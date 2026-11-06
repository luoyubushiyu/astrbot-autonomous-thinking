"""测试包的公共初始化：屏蔽日志输出，避免测试刷屏。"""

import logging

logging.getLogger("autonomous_agent").addHandler(logging.NullHandler())