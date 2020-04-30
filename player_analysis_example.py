import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import matplotlib.pyplot as plt
from PlayerPitchControlAnalysis import PlayerPitchControlAnalysisPlayer

DATADIR = "/users/andrewpuopolo/sample-data/data"
# DATADIR = "WHERE/YOU/STORE/METRICA/DATA"
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


example_player_analysis = PlayerPitchControlAnalysisPlayer(
    tracking_home=tracking_home,
    tracking_away=tracking_away,
    params=params,
    events=events,
    event_id=820,
    team_player_to_analyze="Away",
    player_to_analyze=19,
    field_dimens=(106.0, 68.0),
    n_grid_cells_x=50,
)


# print(
#     example_player_analysis.team_player_to_analyze
#     + " Player "
#     + str(example_player_analysis.player_to_analyze)
#     + " created "
#     + str(int(example_player_analysis.calculate_space_created()))
#     + " m^2 of space with his movement during event "
#     + str(example_player_analysis.event_id)
# )

example_player_analysis.plot_pitch_control_difference(
    replace_function="location",
    relative_x_change=0,
    relative_y_change=20,
    replace_x_velocity=0,
    replace_y_velocity=0,
)
plt.show()
