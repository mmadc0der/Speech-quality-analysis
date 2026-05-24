from __future__ import annotations

import argparse

from pronunciation_backend.services.phone_ssl_pooling import (
    PoolingMode,
    SSL_BASE_DIM,
    resolved_acoustic_input_dim,
)

V3_SSL_FEATURE_FACTOR = 2
V3_D_MODEL = 512
V3_NUM_HEADS = 8
V3_ACOUSTIC_LAYERS = 5
V3_SCORER_LAYERS = 2
V3_FFN_DIM = 2048
V3_POOLING_MODE: PoolingMode = "subspan_end_concat"

V2_DEFAULT_D_MODEL = 384
V2_DEFAULT_NUM_HEADS = 6
V2_DEFAULT_ACOUSTIC_LAYERS = 6
V2_DEFAULT_FFN_DIM = 1_536
V2_DEFAULT_ACOUSTIC_INPUT_DIM = SSL_BASE_DIM


def _resolve_input_dim(args: argparse.Namespace) -> None:
    expected_input = resolved_acoustic_input_dim(ssl_feature_factor=args.ssl_feature_factor)
    if hasattr(args, "acoustic_input_dim"):
        if args.acoustic_input_dim == V2_DEFAULT_ACOUSTIC_INPUT_DIM and args.ssl_feature_factor != 1:
            args.acoustic_input_dim = expected_input
        elif args.acoustic_input_dim != expected_input:
            raise ValueError(
                f"acoustic_input_dim={args.acoustic_input_dim} does not match "
                f"768 * ssl_feature_factor={args.ssl_feature_factor} ({expected_input})."
            )
        return

    if args.input_dim == V2_DEFAULT_ACOUSTIC_INPUT_DIM and args.ssl_feature_factor != 1:
        args.input_dim = expected_input
    elif args.input_dim != expected_input:
        raise ValueError(
            f"input_dim={args.input_dim} does not match "
            f"768 * ssl_feature_factor={args.ssl_feature_factor} ({expected_input})."
        )


def apply_v3_training_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "ssl_feature_factor", None) is None:
        args.ssl_feature_factor = V3_SSL_FEATURE_FACTOR if args.architecture_version == "v3" else 1

    if getattr(args, "pooling_mode", None) is None:
        args.pooling_mode = V3_POOLING_MODE if args.architecture_version == "v3" else "mean"

    _resolve_input_dim(args)

    if args.architecture_version != "v3":
        return

    if args.d_model == V2_DEFAULT_D_MODEL:
        args.d_model = V3_D_MODEL
    if args.num_heads == V2_DEFAULT_NUM_HEADS:
        args.num_heads = V3_NUM_HEADS
    if getattr(args, "acoustic_layers", None) == V2_DEFAULT_ACOUSTIC_LAYERS:
        args.acoustic_layers = V3_ACOUSTIC_LAYERS
    if getattr(args, "num_layers", None) == V2_DEFAULT_ACOUSTIC_LAYERS:
        args.num_layers = V3_ACOUSTIC_LAYERS
    if args.ffn_dim == V2_DEFAULT_FFN_DIM:
        args.ffn_dim = V3_FFN_DIM
