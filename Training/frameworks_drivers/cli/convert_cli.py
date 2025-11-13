from __future__ import annotations

import argparse
from pathlib import Path

from Training.domain.entities import ConversionRequest
from Training.frameworks_drivers.logging import StructuredCsvLogger, get_csv_logger
from Training.interface_adapters.controllers import conversion_controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Detect model formats and run conversion helpers.')
    parser.add_argument('--detect', help='Path to inspect for GGUF/GGML/GPTQ artifacts.')
    parser.add_argument('--convert-llama-ggml-to-hf', action='store_true', help='Execute the Transformers GGML→HF converter (requires --yes).')
    parser.add_argument('--auto-gptq-suggest', action='store_true', help='Show example auto-gptq commands.')
    parser.add_argument('--auto-gptq-convert', action='store_true', help='Run auto-gptq conversion (requires --yes).')
    parser.add_argument('--llama-cpp-suggest', action='store_true', help='Show llama.cpp manual conversion hints.')
    parser.add_argument('--input-dir', help='Input directory for conversion actions.')
    parser.add_argument('--output-dir', help='Output directory for conversion actions.')
    parser.add_argument('--model-size', help='Optional model size hint for the LLaMA converter (e.g. 1B, 7B).')
    parser.add_argument('--llama-version', help='Optional LLaMA version hint (e.g. 3.2).')
    parser.add_argument('--device', default='cpu', help='Device argument forwarded to auto_gptq (default: cpu).')
    parser.add_argument('--dry-run', action='store_true', help='Preview conversion command without executing it.')
    parser.add_argument('--yes', action='store_true', help='Confirm execution of converters.')
    return parser


def _ensure_paths(namespace: argparse.Namespace, logger: StructuredCsvLogger) -> tuple[Path, Path]:
    if not namespace.input_dir or not namespace.output_dir:
        logger.error('Missing required paths', input_dir=namespace.input_dir, output_dir=namespace.output_dir)
        raise ValueError('Please specify --input-dir and --output-dir')
    return Path(namespace.input_dir), Path(namespace.output_dir)


def handle_detection(path: str, logger: StructuredCsvLogger) -> int:
    logger.info('detect_formats_requested', path=path)
    try:
        messages = conversion_controller.detect_formats(Path(path))
        for message in messages:
            print(message)
        logger.info('detect_formats_completed', path=path, detected=len(messages))
        return 0
    except Exception as exc:  # pragma: no cover - defensive catch for CLI surface
        logger.exception('detect_formats_failed', path=path, error=str(exc))
        print('Failed to detect formats:', exc)
        return 1


def handle_llama_conversion(args: argparse.Namespace, logger: StructuredCsvLogger) -> int:
    input_dir, output_dir = _ensure_paths(args, logger)
    logger.info(
        'llama_conversion_requested',
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        dry_run=bool(args.dry_run),
        confirmed=bool(args.yes),
    )
    if not input_dir.exists():
        print('Input directory does not exist:', input_dir)
        logger.error('llama_conversion_missing_input', input_dir=str(input_dir))
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Planned Transformers LLaMA GGML→HF conversion (requires --yes to execute).')
    if args.dry_run:
        print('Dry-run: not executing converter.')
        logger.info('llama_conversion_dry_run', input_dir=str(input_dir), output_dir=str(output_dir))
        return 0
    if not args.yes:
        print('Safety: pass --yes to run the converter.')
        logger.warning('llama_conversion_not_confirmed', input_dir=str(input_dir))
        return 2
    request = ConversionRequest(
        input_dir=input_dir,
        output_dir=output_dir,
        model_size=args.model_size,
        llama_version=args.llama_version,
        dry_run=False,
        confirm=True,
    )
    success = conversion_controller.run_llama_conversion(request)
    if success:
        print('Converter completed; output available at', output_dir)
        logger.info('llama_conversion_completed', input_dir=str(input_dir), output_dir=str(output_dir))
        return 0
    print('Converter failed or script not available.')
    logger.error('llama_conversion_failed', input_dir=str(input_dir), output_dir=str(output_dir))
    return 1


def handle_auto_gptq_suggest(args: argparse.Namespace, logger: StructuredCsvLogger) -> int:
    input_dir, output_dir = _ensure_paths(args, logger)
    print('\nSuggested auto-gptq workflow (manual):')
    print('  python -m auto_gptq.convert --model_name_or_path', input_dir, '--output_dir', output_dir, '--device cpu')
    print('  # Validate the converted checkpoint with Training/tools/check_load_model.py')
    logger.info('auto_gptq_suggestion_rendered', input_dir=str(input_dir), output_dir=str(output_dir))
    return 0


def handle_auto_gptq_convert(args: argparse.Namespace, logger: StructuredCsvLogger) -> int:
    input_dir, output_dir = _ensure_paths(args, logger)
    logger.info(
        'auto_gptq_conversion_requested',
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        dry_run=bool(args.dry_run),
        confirmed=bool(args.yes),
        device=args.device,
    )
    if not input_dir.exists():
        print('Input directory does not exist:', input_dir)
        logger.error('auto_gptq_missing_input', input_dir=str(input_dir))
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    print('Planned auto-gptq conversion (requires auto_gptq installed).')
    if args.dry_run:
        print('Dry-run: not executing auto-gptq conversion.')
        logger.info('auto_gptq_conversion_dry_run', input_dir=str(input_dir), output_dir=str(output_dir))
        return 0
    if not args.yes:
        print('Safety: pass --yes to run the auto-gptq converter.')
        logger.warning('auto_gptq_not_confirmed', input_dir=str(input_dir))
        return 2
    request = ConversionRequest(
        input_dir=input_dir,
        output_dir=output_dir,
        device=args.device,
        dry_run=False,
        confirm=True,
    )
    success = conversion_controller.run_auto_gptq_conversion(request)
    if success:
        print('auto-gptq conversion completed; output available at', output_dir)
        logger.info('auto_gptq_conversion_completed', input_dir=str(input_dir), output_dir=str(output_dir))
        return 0
    print('auto-gptq conversion failed or auto_gptq is unavailable.')
    logger.error('auto_gptq_conversion_failed', input_dir=str(input_dir), output_dir=str(output_dir))
    return 1


def handle_llama_cpp_hint(args: argparse.Namespace, logger: StructuredCsvLogger) -> int:
    input_dir, output_dir = _ensure_paths(args, logger)
    print('\nSuggested llama.cpp workflow (manual):')
    print('  python lib\\llama.cpp\\convert_hf_to_gguf.py', input_dir, '--outfile <out.gguf> --outtype f16')
    print('  # Reverse conversions (GGUF/GGML → HF) rely on community scripts; consult the model card.')
    logger.info('llama_cpp_hint_rendered', input_dir=str(input_dir), output_dir=str(output_dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = get_csv_logger('convert_cli')
    logger.info('convert_cli_invoked', argv=list(argv) if argv is not None else [])
    exit_code = 0
    try:
        if args.detect:
            exit_code = handle_detection(args.detect, logger)
        elif args.convert_llama_ggml_to_hf:
            exit_code = handle_llama_conversion(args, logger)
        elif args.auto_gptq_suggest:
            exit_code = handle_auto_gptq_suggest(args, logger)
        elif args.auto_gptq_convert:
            exit_code = handle_auto_gptq_convert(args, logger)
        elif args.llama_cpp_suggest:
            exit_code = handle_llama_cpp_hint(args, logger)
        else:
            parser.print_help()
            exit_code = 0
        logger.info('convert_cli_completed', exit_code=exit_code)
        return exit_code
    except Exception as exc:  # pragma: no cover - defensive top-level guard
        logger.exception('convert_cli_unhandled_exception', error=str(exc))
        print('Unexpected error:', exc)
        return 1
    finally:
        logger.close()


if __name__ == '__main__':
    raise SystemExit(main())
