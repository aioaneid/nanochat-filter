#define LTV_LOOK_BACK_LAUNCH_EACH_PQK()  \
    LAUNCH_IF(2147483647, 1, 1, true);  \
    LAUNCH_IF(16, 16, 9, true);         \
                                        \
    /* Test-only configurations */      \
    LAUNCH_IF(2147483647, 1, 1, false); \
    LAUNCH_IF(1024, 1024, 1, false);    \
    LAUNCH_IF(64, 60, 5, false);        \
    LAUNCH_IF(128, 112, 17, false);     \
    LAUNCH_IF(160, 108, 53, false);     \
    LAUNCH_IF(192, 104, 89, false);     \
    LAUNCH_IF(224, 100, 125, false)
