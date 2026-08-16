"""Print a sanitized, offline DataChef configuration diagnostic."""

from dotenv import load_dotenv

from utils.runtime_config import inspect_runtime_configuration


def main() -> int:
    load_dotenv(override=False)
    check = inspect_runtime_configuration()
    print(check.model_dump_json(indent=2))
    return 0 if check.provider_status.value != "INVALID_CONFIGURATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
