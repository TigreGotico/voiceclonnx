# START_VERSION_BLOCK
VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_BUILD = 1
VERSION_ALPHA = 1
# END_VERSION_BLOCK

# __version__ is derived from the block above so that the release automation
# (which only rewrites the block) propagates to the package version.
VERSION_TUPLE = (VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD)
__version__ = ".".join(str(n) for n in VERSION_TUPLE)
if VERSION_ALPHA:
    __version__ += f"a{VERSION_ALPHA}"
