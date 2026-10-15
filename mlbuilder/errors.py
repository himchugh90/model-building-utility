"""Exception hierarchy. Every user-facing failure derives from MlBuilderError."""


class MlBuilderError(Exception):
    """Base class for all errors raised deliberately by the utility."""


class ConfigError(MlBuilderError):
    """The configuration is invalid (bad schema, unknown component, bad params)."""


class DataError(MlBuilderError):
    """The data does not satisfy what the configuration expects."""
