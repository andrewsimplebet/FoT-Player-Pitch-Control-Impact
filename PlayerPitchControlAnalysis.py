import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import Metrica_Viz as mviz
import numpy as np
import matplotlib.pyplot as plt

DATADIR = "/users/andrewpuopolo/sample-data/data"
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


class PlayerPitchControlAnalysisPlayer(object):
    def __init__(
        self,
        tracking_home,
        tracking_away,
        params,
        events,
        event_id,
        team_player_to_analyze,
        player_to_analyze,
        field_dimens=(106.0, 68.0),
        n_grid_cells_x=50,
    ):
        """
        This class is used to consolidate many of the functions that would be used to analyze the impact of a player's movement on pitch control
        Leveraging @EightyFivePoint's pitch control model as presented in the Friends of Tracking Series, we build out a series of tools to help isolate individual player's impacts to pitch control
        Using an event from the match's event dataframe, and a specific team/player ID, we can:
            1. Calculate the amount of space occupied on the pitch (per EightyFivePoint's pitch control model) for any frame of a match
            2. Calculate the difference in total space occupied by the player's team with his current movement vs theoretical movement
            3. Compute the pitch control surface if a specific player had a different velocity vector
            4. Compute the difference in the pitch control surface between a player's actual velocity vector, and a theorized one
            5. Plot the space created (and lost) by a player's actual movement

        :param pd.DataFrame tracking_home : tracking DataFrame for the Home team, containing velocity vectors for each player
        :param pd.DataFrame tracking_away: tracking DataFrame for the Away team, containing velocity vectors for each player
        :param dict params: Dictionary of model parameters (default model parameters can be generated using default_model_params())
        :param pd.DataFrame events: Dataframe containing the event data
        :param int event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        :param str team_player_to_analyze: The team of the player whose movement we want to analyze. Must be either "Home" or "Away"
        :param int or str(int) player_to_analyze: The player ID of the player whose movement we want to analyze.
        :param tuple field_dimens: tuple containing the length and width of the pitch in meters. Default is (106,68)
        :param int n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        """
        self.tracking_home = tracking_home
        self.tracking_away = tracking_away
        self.params = params
        self.events = events
        self.event_id = event_id
        self.team_player_to_analyze = team_player_to_analyze
        self.player_to_analyze = player_to_analyze
        self.field_dimens = field_dimens
        self.n_grid_cells_x = n_grid_cells_x

    def calculate_total_space_on_pitch_team(self, pitch_control_result, xgrid, ygrid):
        """
        This function calculates the number of square meters on the pitch occupied by the team with the ball in the given event.

        :param numpy.ndarray pitch_control_result: The estimates from the result of a fitted pitch control model
        :param numpy.ndarray xgrid: The xgrid from the result of a fitted pitch control model
        :param numpy.ndarray ygrid: The ygrid from the result of a fitted pitch control model
        :return: The number of meters occupied by the attacking team in a freeze frame of the data. Measured in m^2
        """

        total_space_attacking = (
            self.field_dimens[0]
            * self.field_dimens[1]
            * (pitch_control_result.sum())
            / (len(xgrid) * len(ygrid))
        )
        return total_space_attacking

    def calculate_pitch_control_replaced_player(
        self, replace_x_velocity=0, replace_y_velocity=0,
    ):
        """
        This function calculates a pitch control surface after replacing a player's velocity vector with a new one

        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with
        :return
        edited_pitch_control: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team with one player's velocity changed
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
        """
        self._validate_inputs()

        event_frame = self.events.loc[self.event_id]["Start Frame"]

        # Replace datapoints with 0 vectors
        temp_tracking_home = self.tracking_home.copy()
        temp_tracking_away = self.tracking_away.copy()

        if self.team_player_to_analyze == "Home":

            temp_tracking_home["Home_" + str(self.player_to_analyze) + "_vx"].loc[
                event_frame
            ] = replace_x_velocity
            temp_tracking_home["Home_" + str(self.player_to_analyze) + "_vy"].loc[
                event_frame
            ] = replace_y_velocity
        elif self.team_player_to_analyze == "Away":
            temp_tracking_away["Away_" + str(self.player_to_analyze) + "_vx"].loc[
                event_frame
            ] = replace_x_velocity
            temp_tracking_away["Away_" + str(self.player_to_analyze) + "_vy"].loc[
                event_frame
            ] = replace_y_velocity

        edited_pitch_control, xgrid, ygrid = mpc.generate_pitch_control_for_event(
            event_id=self.event_id,
            events=self.events,
            tracking_home=temp_tracking_home,
            tracking_away=temp_tracking_away,
            params=self.params,
            field_dimen=self.field_dimens,
            n_grid_cells_x=self.n_grid_cells_x,
        )
        return edited_pitch_control, xgrid, ygrid

    def calculate_pitch_control_difference(
        self, replace_x_velocity=0, replace_y_velocity=0
    ):
        """
        This function computes the difference in pitch control surfaces between the actual event, and the event if one player's velocity vector is altered

        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with
        :return:
            pitch_control_difference: Difference in pitch control surfacesPitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) between when the relevant player uses his actual movement and his theorized movement
            xgrid: Positions of the pixels in the x-direction (field length)
            ygrid: Positions of the pixels in the y-direction (field width)
        """
        actual_pitch_control, xgrid, ygrid = mpc.generate_pitch_control_for_event(
            event_id=self.event_id,
            events=self.events,
            tracking_home=self.tracking_home,
            tracking_away=self.tracking_away,
            params=self.params,
        )
        (
            edited_pitch_control,
            xgrid,
            ygrid,
        ) = self.calculate_pitch_control_replaced_player(
            replace_x_velocity=replace_x_velocity,
            replace_y_velocity=replace_y_velocity,
        )
        pitch_control_difference = actual_pitch_control - edited_pitch_control
        return pitch_control_difference, xgrid, ygrid

    def calculate_space_created(self, replace_x_velocity=0, replace_y_velocity=0):
        """
        This function calculates the total amount of space generated by a player's movement, relative to a theoretical velocity vector

        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with
        :return: A float representing amount of pitch gained (or lost) by the player's current movement (versus alternative movement). Measured in m^2
        """
        team_with_possession = self.events.loc[self.event_id].Team

        (
            pitch_control_difference,
            xgrid,
            ygrid,
        ) = self.calculate_pitch_control_difference(
            replace_x_velocity=replace_x_velocity, replace_y_velocity=replace_y_velocity
        )

        pitch_control_change = self.calculate_total_space_on_pitch_team(
            pitch_control_difference, xgrid, ygrid
        )
        if team_with_possession == self.team_player_to_analyze:
            return pitch_control_change
        else:
            return -1 * pitch_control_change

    def plot_pitch_control_difference(self, replace_x_velocity=0, replace_y_velocity=0):
        """
        This function provides a plot for the space created (and lost) by a player's movement

        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with

        """
        (
            pitch_control_difference,
            xgrid,
            ygrid,
        ) = self.calculate_pitch_control_difference(
            replace_x_velocity=replace_x_velocity, replace_y_velocity=replace_y_velocity
        )

        mviz.plot_pitchcontrol_for_event(
            event_id=self.event_id,
            events=self.events,
            tracking_home=self.tracking_home,
            tracking_away=self.tracking_away,
            PPCF=pitch_control_difference,
            annotate=True,
            xgrid=xgrid,
            ygrid=ygrid,
            plotting_difference=True,
        )
        plt.title(
            "Space created by "
            + str(self.team_player_to_analyze)
            + " Player "
            + str(self.player_to_analyze)
            + " during event "
            + str(self.event_id),
            fontdict={"fontsize": 22},
        )

    def get_players_on_pitch(self):
        pass_frame = self.events.loc[self.event_id]["Start Frame"]
        players_on_pitch = []
        if self.team_player_to_analyze == "Home":
            data_row = self.tracking_home.loc[pass_frame]
        else:
            data_row = self.tracking_away.loc[pass_frame]
        for index in data_row.index:
            if "_vx" in index:
                if not np.isnan(data_row.loc[index]):
                    players_on_pitch.append(index.split("_")[1])
        return players_on_pitch

    def _validate_inputs(self):
        if type(self.player_to_analyze) not in (str, int):
            raise ValueError("player_to_analyze must be an integer or a string")

        if self.team_player_to_analyze not in ["Home", "Away"]:
            raise ValueError("team_player_to_analyze must equal 'Home' or 'Away'")

        if str(self.player_to_analyze) not in self.get_players_on_pitch():
            raise ValueError(
                "player_to_analyze is either not on the correct team, or was not on the pitch at the time of the event"
            )


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


print(example_player_analysis.team_player_to_analyze
      + " Player "
      + str(example_player_analysis.player_to_analyze)
      + " created " + str(int(example_player_analysis.calculate_space_created()))
      + " m^2 of space with his movement during event "
      + str(example_player_analysis.event_id)
      )

example_player_analysis.plot_pitch_control_difference()
plt.savefig('FoT_player19.png')