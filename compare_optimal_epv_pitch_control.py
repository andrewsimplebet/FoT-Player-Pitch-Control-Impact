"""
In this file, we look at Player 5 from Event 820, and compare how the "optimal location" differs between using
pitch control (assumes all locations on pitch are equal) and EPV weighted pitch control (which weights more dangerous
areas more)
"""

import data_setup as data
from PlayerEventAnalysis import PlayerEventAnalysis

common_kw_args = {
    "tracking_home": data.tracking_home,
    "tracking_away": data.tracking_away,
    "params": data.params,
    "events": data.events,
    "event_id": 820,
    "gk_numbers": data.GK_numbers,
    "field_dimens": (106.0, 68.0),
    "n_grid_cells_x": 50,
}

example_player_analysis_pitch_control = PlayerEventAnalysis(
    **common_kw_args, team_player_to_analyze="Home", player_to_analyze=5, epv=False
)

example_player_analysis_pitch_control.get_optimal_location_on_pitch(
    size_of_grid=20, location_trials=125, velocity_trials=30, max_velocity=5
)


example_player_analysis_epv = PlayerEventAnalysis(
    **common_kw_args, team_player_to_analyze="Home", player_to_analyze=5, epv=True
)

example_player_analysis_epv.get_optimal_location_on_pitch(
    size_of_grid=20, location_trials=125, velocity_trials=30, max_velocity=5
)
