"""
In this file, we take a look at our metrics and plots for Player 23, using EPV as our surface
"""

import data_setup as data
import matplotlib.pyplot as plt
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

epv_player_analysis = PlayerEventAnalysis(
    **common_kw_args, team_player_to_analyze="Away", player_to_analyze=23, epv=True
)

epv_player_analysis.plot_pitch_control_difference(
    replace_function="movement", replace_x_velocity=0, replace_y_velocity=0
)
plt.show()


print(
    epv_player_analysis.team_player_to_analyze
    + " Player "
    + str(epv_player_analysis.player_to_analyze)
    + " occupied "
    + str(
        round(
            100*epv_player_analysis.calculate_space_created(
                replace_function="presence"
            ), 2
        )
    )
    + " percent of the EPV surface during event "
    + str(epv_player_analysis.event_id)
)

# And plot this space on the pitch:

epv_player_analysis.plot_pitch_control_difference(replace_function="presence")
plt.show()


print(
    epv_player_analysis.team_player_to_analyze
    + " Player "
    + str(epv_player_analysis.player_to_analyze)
    + " would have occupied a difference of "
    + str(
        round(
            -100
            * epv_player_analysis.calculate_space_created(
                replace_function="location", relative_x_change=3, relative_y_change=0
            ),
            2,
        )
    )
    + " percent of the EPV surface for his team "
    + str(epv_player_analysis.event_id)
    + " if they were 5 meters further forwards"
)


epv_player_analysis.plot_pitch_control_difference(
    replace_function="location", relative_x_change=3, relative_y_change=0
)
plt.show()
