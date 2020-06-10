from enum import Enum


class MultiViewEncoderType(Enum):
    Pool = 0
    Stack = 1


def fetch_multi_view_encoder_type(type_str: str) -> MultiViewEncoderType:
    if type_str == 'pool':
        return MultiViewEncoderType.Pool
    elif type_str == 'stack':
        return MultiViewEncoderType.Stack
    else:
        raise TypeError('invalid multi view encoder type: {}'.format(type_str))
