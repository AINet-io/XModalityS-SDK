# © Copyright 2023, Liu Yuchen, All rights reserved.

from enum import IntEnum


class XModalitySCode(IntEnum):
    Success = 0
    ExternalError = 1
    ModelhubError = 2
    DatahubError = 3
    Timeout = 4


class XModalitySError(RuntimeError):
    def __init__(self, error: str):
        self.error = error

    def __str__(self):
        return self.error


class XModalitySInternalError(XModalitySError):
    def __init__(self, error: str | None = None):
        XModalitySError.__init__(self, error)


class XModalitySExternalError(XModalitySError):
    def __init__(self, error: str | None = None):
        XModalitySError.__init__(self, error)
