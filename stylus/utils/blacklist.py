ADAPTER_BLACKLIST = [
    87182,
    204617,
    39123,
    137134,
    313839,
    220074,
    81946,
    89385,
    74524,
    96573,
    59264,
    70602,
    79374,
    80713,
    70723,
    70712,
    80719,
    69013,
    71682,
    69026,
    77373,
    29170,
    89322,
    70296,
    79036,
    159307,
    347694,
    107319,
    68462,
    245115,
    54581,
    68423,
    148099,
    337139,
    121499,
    84879,
    216624,
    75034,
    139923,
    90025,
    137904,
    138839,
    137448,
    139465,
    137344,
    138985,
    273899,
    84235,
    79955,
    303325,
    97200,
    242426,
    330759,
    89697,
    284008,
    240522,
    82582,
    254714,
    176442,
]


def blacklist_adapters(adapters, enable_characters=False):
    """Filter out adapters based on various criteria to improve image quality.

    Args:
        adapters (list): List of adapter instances to filter.
        enable_characters (bool): Flag to determine if character-related adapters should be included.

    Returns:
        list: Filtered list of adapters.
    """
    # Adapters that generally produce very bad images.
    adapters = [a for a in adapters if a.adapter_id not in ADAPTER_BLACKLIST]

    # Adapters which have too few images - hard to know what they do in practice.
    adapters = [a for a in adapters if len(a.image_urls) > 2]

    # Concept sliders. Their weight range is too dynamic. Also blacklist weird DND maps.
    adapters = [
        a for a in adapters if 'slider' not in a.title.lower()
        and 'table rpg' not in a.title.lower()
    ]

    # Optionally blacklist character/celebrity adapters to improve performance on specific datasets.
    if not enable_characters:
        adapters = [
            a for a in adapters
            if 'character' not in a.tags and 'celebrity' not in a.tags
        ]

    return adapters
