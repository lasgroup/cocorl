# type: ignore
"""Helper functions for logging experiments."""
import datetime

import sacred


# changes the run _id and thereby the path that the FileStorageObserver
# writes the results
# cf. https://github.com/IDSIA/sacred/issues/174
class SetID(sacred.observers.RunObserver):
    """Set the experiment folder ID."""

    priority = 50  # very high priority to set id

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        custom_id = "{}".format(timestamp)
        return custom_id  # started_event returns the _run._id
