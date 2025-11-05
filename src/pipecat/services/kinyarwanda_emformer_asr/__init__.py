#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys

from pipecat.services import DeprecatedModuleProxy

from .stt import *

sys.modules[__name__] = DeprecatedModuleProxy(globals(), "kinyarwanda_emformer_asr", "kinyarwanda_emformer_asr.stt")
