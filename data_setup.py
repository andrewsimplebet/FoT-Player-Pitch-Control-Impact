"""
In this file, we store all the relevant information for use in our examples in other files. We put this information
here in order to reduce the amount of code that must be written elsewhere, and share it to other files.
"""

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc

DATADIR = "/users/andrewpuopolo/sample-data/data"
# DATADIR = "WHERE/YOU/STORE/FREE/METRICA/DATA"

# region Laurie's code
game_id = 2  # let's look at sample match 2

# read in the event data
events = mio.read_event_data(DATADIR, game_id)

# read in tracking data
tracking_home = mio.tracking_data(DATADIR, game_id, "Home")
tracking_away = mio.tracking_data(DATADIR, game_id, "Away")

# Convert positions from metrica units to meters (note change in Metrica's coordinate system since the last lesson)
tracking_home = mio.to_metric_coordinates(tracking_home)
tracking_away = mio.to_metric_coordinates(tracking_away)
events = mio.to_metric_coordinates(events)

# reverse direction of play in the second half so that home team is always attacking from right->left
tracking_home, tracking_away, events = mio.to_single_playing_direction(
    tracking_home, tracking_away, events
)

# Calculate player velocities
tracking_home = mvel.calc_player_velocities(tracking_home, smoothing=True)
tracking_away = mvel.calc_player_velocities(tracking_away, smoothing=True)
params = mpc.default_model_params(3)
# endregion

# Get GK numbers
GK_numbers = [mio.find_goalkeeper(tracking_home), mio.find_goalkeeper(tracking_away)]
