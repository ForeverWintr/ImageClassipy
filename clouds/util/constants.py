import enum

import camel

healthStatusRegistry = camel.CamelRegistry()


class HealthStatus(enum.IntEnum):
    GOOD = 0
    CLOUDY = 1
    CANOLA = 2
    INSUFFICIENT_COVERAGE = 3
    REJECTED_OTHER = 4


@healthStatusRegistry.dumper(HealthStatus, 'HealthStatus', 1)
def _dumpHealthStatus(obj):
    return {
        "status": obj.name,
    }


@healthStatusRegistry.loader('HealthStatus', 1)
def _loadHealthStatus(data, version):
    return HealthStatus._member_map_[data['status']]
