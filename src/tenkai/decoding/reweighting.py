"""Tenkai exclusive reweighting with per-lifecycle hybrid routing."""

from __future__ import annotations

from collections import OrderedDict
from math import isfinite
from typing import Literal, Mapping, Sequence

import numpy as np
import pymatching
import stim

from tenkai.circuits.metadata import CircuitMetadata
from tenkai.decoding.activation import LossShadowAtlasBuilder
from tenkai.decoding.fusion import (
    ActivationMap,
    backsolve_xor_source_map,
    build_reweighted_matcher,
    combine_xor_multi,
)
from tenkai.decoding.lossy_dem import LossyDemDecoder


LifecycleStrategy = Literal["reweight", "lossy_dem"]


class TenkaiDecoder:
    """Decode visible loss with Tenkai's exclusive-reweighting graph."""

    def __init__(
        self,
        *,
        clean_circuit: stim.Circuit,
        dem_circuit: stim.Circuit,
        metadata: CircuitMetadata,
        loss_probability: float,
        strategy_by_lifecycle: Mapping[int, LifecycleStrategy | str],
        rho_by_lifecycle: Mapping[int, float],
        lossy_dem_decoder: LossyDemDecoder | None = None,
        matcher_cache_size: int = 4096,
    ) -> None:
        if type(matcher_cache_size) is not int or matcher_cache_size < 1:
            raise ValueError(
                "matcher_cache_size must be a positive integer, "
                f"got {matcher_cache_size!r}"
            )
        if type(loss_probability) not in {int, float}:
            raise ValueError(
                "loss_probability must be a finite number in [0, 1], "
                f"got {loss_probability!r}"
            )
        normalized_loss_probability = float(loss_probability)
        if (
            not isfinite(normalized_loss_probability)
            or not 0.0 <= normalized_loss_probability <= 1.0
        ):
            raise ValueError(
                "loss_probability must be a finite number in [0, 1], "
                f"got {loss_probability!r}"
            )
        self.metadata = metadata
        self.loss_probability = normalized_loss_probability
        self.lossy_dem_decoder = lossy_dem_decoder or LossyDemDecoder(
            dem_circuit=dem_circuit,
            metadata=metadata,
            loss_probability=loss_probability,
        )
        if self.lossy_dem_decoder.metadata.circuit_sha256 != metadata.circuit_sha256:
            raise ValueError("lossy_dem_decoder metadata does not match Tenkai metadata")
        if self.lossy_dem_decoder.loss_probability != self.loss_probability:
            raise ValueError(
                "lossy_dem_decoder loss probability does not match Tenkai: "
                f"decoder={self.lossy_dem_decoder.loss_probability}, "
                f"tenkai={self.loss_probability}"
            )

        self.clean_matcher = self.lossy_dem_decoder.clean_matcher
        self.num_detectors = int(dem_circuit.num_detectors)
        self.num_observables = int(dem_circuit.num_observables)
        self.atlas = LossShadowAtlasBuilder(
            clean_circuit=clean_circuit,
            metadata=metadata,
            loss_probability=loss_probability,
        )
        self.strategy_by_lifecycle, self.rho_by_lifecycle = (
            self._validate_routing(
                strategy_by_lifecycle,
                rho_by_lifecycle,
            )
        )
        self._source_map_cache: dict[int, ActivationMap] = {}
        self._matcher_cache: OrderedDict[
            tuple[int, ...], pymatching.Matching
        ] = OrderedDict()
        self._matcher_cache_size = matcher_cache_size

    def _validate_routing(
        self,
        raw_strategies: Mapping[int, LifecycleStrategy | str],
        raw_rhos: Mapping[int, float],
    ) -> tuple[dict[int, LifecycleStrategy], dict[int, float]]:
        strategies: dict[int, LifecycleStrategy] = {}
        for raw_key, raw_strategy in raw_strategies.items():
            if type(raw_key) is not int or raw_key < 0:
                raise ValueError(
                    "strategy lifecycle keys must be non-negative exact integers, "
                    f"got {raw_key!r}"
                )
            if raw_key in strategies:
                raise ValueError(f"duplicate strategy lifecycle key: {raw_key}")
            if raw_strategy not in {"reweight", "lossy_dem"}:
                raise ValueError(
                    "lifecycle strategy must be 'reweight' or 'lossy_dem', "
                    f"got lifecycle={raw_key}, strategy={raw_strategy!r}"
                )
            normalized_strategy: LifecycleStrategy = (
                "reweight" if raw_strategy == "reweight" else "lossy_dem"
            )
            strategies[raw_key] = normalized_strategy

        expected = set(range(len(self.atlas.lifecycles)))
        supplied = set(strategies)
        if supplied != expected:
            raise ValueError(
                "Tenkai routing lifecycle coverage mismatch: "
                f"missing={sorted(expected - supplied)[:3]!r}, "
                f"extra={sorted(supplied - expected)[:3]!r}"
            )

        rhos: dict[int, float] = {}
        for raw_key, raw_rho in raw_rhos.items():
            if type(raw_key) is not int or raw_key < 0:
                raise ValueError(
                    "rho lifecycle keys must be non-negative exact integers, "
                    f"got {raw_key!r}"
                )
            if type(raw_rho) not in {int, float}:
                raise ValueError(
                    "rho must be a finite number in (0, 1], "
                    f"got lifecycle={raw_key}, rho={raw_rho!r}"
                )
            rho = float(raw_rho)
            if not isfinite(rho) or not 0.0 < rho <= 1.0:
                raise ValueError(
                    "rho must be a finite number in (0, 1], "
                    f"got lifecycle={raw_key}, rho={raw_rho!r}"
                )
            if raw_key in rhos:
                raise ValueError(f"duplicate rho lifecycle key: {raw_key}")
            rhos[raw_key] = rho

        expected_rhos = {
            key for key, strategy in strategies.items() if strategy == "reweight"
        }
        if set(rhos) != expected_rhos:
            raise ValueError(
                "rho lifecycle keys must exactly match reweight routes: "
                f"missing={sorted(expected_rhos - set(rhos))[:3]!r}, "
                f"extra={sorted(set(rhos) - expected_rhos)[:3]!r}"
            )
        return strategies, rhos

    def lifecycle_ids_for_measurements(
        self,
        lost_measurement_indices: Sequence[int],
    ) -> tuple[int, ...]:
        """Resolve visible measurements to sorted unique atlas lifecycle IDs."""

        lifecycle_ids: set[int] = set()
        for raw_index in lost_measurement_indices:
            if type(raw_index) is not int:
                raise ValueError(
                    "lost measurement indices must be exact integers, "
                    f"got {raw_index!r}"
                )
            try:
                lifecycle_id = self.atlas.measurement_to_lifecycle_id[raw_index]
            except KeyError as exc:
                raise ValueError(
                    "lost measurement has no Tenkai lifecycle: "
                    f"measurement_index={raw_index}"
                ) from exc
            lifecycle_ids.add(lifecycle_id)
        return tuple(sorted(lifecycle_ids))

    def activation_map(self, lifecycle_id: int) -> ActivationMap:
        """Return one unscaled deterministic loss-shadow activation map."""

        return self.atlas.activation_map(
            lifecycle_id,
            matcher=self.clean_matcher,
        )

    def source_map(self, lifecycle_id: int) -> ActivationMap:
        """Return one routed and rho-scaled XOR source map."""

        lifecycle = self.atlas.lifecycle(lifecycle_id)
        cached = self._source_map_cache.get(lifecycle_id)
        if cached is not None:
            return dict(cached)

        strategy = self.strategy_by_lifecycle[lifecycle_id]
        if strategy == "reweight":
            rho = self.rho_by_lifecycle[lifecycle_id]
            source = {
                edge: min(max(rho * float(probability), 0.0), 1.0)
                for edge, probability in self.activation_map(lifecycle_id).items()
                if rho * float(probability) > 0.0
            }
        else:
            local_dem = self.lossy_dem_decoder.modified_dem(
                [lifecycle.measurement_index]
            )
            local_matcher = pymatching.Matching.from_detector_error_model(local_dem)
            source = backsolve_xor_source_map(
                clean_matcher=self.clean_matcher,
                local_matcher=local_matcher,
            )
        self._source_map_cache[lifecycle_id] = source
        return dict(source)

    def matcher_for_losses(
        self,
        lost_measurement_indices: Sequence[int],
    ) -> pymatching.Matching:
        """Return the cached reweighted clean matcher for one visible loss set."""

        lifecycle_ids = self.lifecycle_ids_for_measurements(
            lost_measurement_indices
        )
        if not lifecycle_ids:
            return self.clean_matcher
        cached = self._matcher_cache.get(lifecycle_ids)
        if cached is not None:
            self._matcher_cache.move_to_end(lifecycle_ids)
            return cached

        source_maps = [
            source
            for lifecycle_id in lifecycle_ids
            if (source := self.source_map(lifecycle_id))
        ]
        if not source_maps:
            matcher = self.clean_matcher
        else:
            matcher = build_reweighted_matcher(
                clean_matcher=self.clean_matcher,
                activation_map=combine_xor_multi(source_maps),
                num_detectors=self.num_detectors,
            )
        self._matcher_cache[lifecycle_ids] = matcher
        self._matcher_cache.move_to_end(lifecycle_ids)
        while len(self._matcher_cache) > self._matcher_cache_size:
            self._matcher_cache.popitem(last=False)
        return matcher

    def decode(
        self,
        detectors: np.ndarray,
        *,
        lost_measurement_indices: Sequence[int],
    ) -> np.ndarray:
        """Decode one detector sample into observable corrections."""

        syndrome = np.asarray(detectors, dtype=np.uint8).reshape(-1)
        if syndrome.size != self.num_detectors:
            raise ValueError(
                f"detector count must be {self.num_detectors}, got {syndrome.size}"
            )
        matcher = self.matcher_for_losses(lost_measurement_indices)
        prediction = matcher.decode(syndrome)
        return np.asarray(prediction, dtype=np.uint8).reshape(-1)
