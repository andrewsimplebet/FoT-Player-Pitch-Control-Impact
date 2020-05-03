import Metrica_PitchControl as mpc
import Metrica_Viz as mviz
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
from hyperopt import hp, fmin, tpe, Trials


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
        This class is used to consolidate many of the functions that would be used to analyze the impact of an
            individual on the pitch control surface during a given event of a match
        Leveraging @EightyFivePoint's pitch control model presented in the Friends of Tracking Series, we build out a
            series of tools to help isolate individual player's impacts to pitch control.
        Using an event from the match's event DataFrame, and a specific team/player ID, this class allows us to:
            1. Calculate the amount of space occupied on the pitch (per EightyFivePoint's pitch control model) for any
                frame of a match.
            2. Calculate the difference in total space occupied by the player's team and plot the difference in pitch
                control surfaces with his/her current movement relative to a theoretical velocity vector
                (which defaults to no movement).
            3. Calculate the difference in total space occupied by the player's team and plot the difference in pitch
                control surfaces relative to if the player were not on the pitch at all.
            4.  Calculate the difference in total space occupied by the player's team and plot the difference in pitch
                control surfaces relative to if the player were in a different location on the pitch and containing a
                new velocity vector

        In the relevant functions for plotting pitch control difference and space creation, the ``replace_function``
        argument determines what type of analysis we wish to carry out.
            If replace_function=``movement``, we study the player's impact on pitch control relative to the pitch
            control if the player had a different velocity vector.
            velocity vector.
            If replace_function=``presence``, we study the player's total impact on pitch control by comparing the
            pitch control surface to the pitch control surface if the player were not on the pitch at all.
            If replace_function=``location``, we study the player's impact on pitch control relative to the pitch
            control if the player were in a different location on the pitch.

        Examples of using this class for each type of analysis are contained in the file ``player_analysis_example.py``.

        Modifications to @EightyFivePoint's code for plotting pitch control to support our new plots can be found in
        ``Metrica_viz.py``

        Initialization parameters:
        :param pd.DataFrame tracking_home : tracking DataFrame for the Home team, containing velocity vectors for each
                player.
        :param pd.DataFrame tracking_away: tracking DataFrame for the Away team, containing velocity vectors for each
                player
        :param dict params: Dictionary of model parameters (default model parameters can be generated using
                default_model_params())
        :param pd.DataFrame events: DataFrame containing the event data
        :param int event_id: Index (not row) of the event that describes the instant at which the pitch control surface
                should be calculated
        :param str team_player_to_analyze: The team of the player whose movement we want to analyze. Must be either
                "Home" or "Away"
        :param int or str(int) player_to_analyze: The player ID of the player whose movement we want to analyze. The ID
                must be a player currently on the pitch for ``team_player_to_analyze``
        :param tuple field_dimens: tuple containing the length and width of the pitch in meters. Default is (106,68)
        :param int n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface.
                Default is 50. n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions


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
        self.tracking_frame = self.events.loc[self.event_id]["Start Frame"]
        self.team_in_possession = self.events.loc[self.event_id]["Team"]
        (
            self.event_pitch_control,
            self.xgrid,
            self.ygrid,
        ) = mpc.generate_pitch_control_for_event(
            event_id=self.event_id,
            events=self.events,
            tracking_home=self.tracking_home,
            tracking_away=self.tracking_away,
            params=self.params,
        )

    def calculate_total_space_on_pitch_team(
        self, pitch_control_result, calculating_diff=False
    ):
        """
        Function Description:
        This function calculates the number of square meters on the pitch occupied by the team with the ball in the
        given event. This assumes that all portions of the pitch are equal in value.

        Input Parameters:
        :param numpy.ndarray pitch_control_result: The estimates from the result of a fitted pitch control model

        Returns:
        :return: The number of meters occupied by the attacking team in a freeze frame of the data. Measured in m^2
        """

        total_space_attacking = (
            self.field_dimens[0]
            * self.field_dimens[1]
            * (pitch_control_result.sum())
            / (len(self.xgrid) * len(self.ygrid))
        )

        if (self.team_player_to_analyze == self.team_in_possession) or calculating_diff:
            return total_space_attacking
        else:
            return (self.field_dimens[0] * self.field_dimens[1]) - total_space_attacking

    def calculate_pitch_control_replaced_velocity(
        self, replace_x_velocity=0, replace_y_velocity=0,
    ):
        """
        Function Description:
        This function calculates a pitch control surface after replacing a player's velocity vector with a new one.
        Ideally, this would be used to determine what space a player is gaining (and conceding) with his/her off the
        ball movement

        Input Parameters:
        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with.
                Positive values will move the player toward's the home team's goal, while negative values will move the
                player towards the away team's goal. Measured in meters per second. Defaults to 0
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with.
                Positive values will move the player to the left side of the pitch, from the perspective of the away
                team, while negative values will move the player towards the right side of the pitch from the
                perspective of the away team. Measured in meters per second. Defaults to 0.

        Returns:
        edited_pitch_control: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control
                probability for the attacking team with one player's velocity changed (defaulted to not moving).
                Surface for the defending team is simply 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
        """
        self._validate_inputs()

        # Replace player's velocity datapoints with new velocity vector
        temp_tracking_home = self.tracking_home.copy()
        temp_tracking_away = self.tracking_away.copy()

        if self.team_player_to_analyze == "Home":
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_vx"
            ] = replace_x_velocity
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_vy"
            ] = replace_y_velocity
        elif self.team_player_to_analyze == "Away":
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_vx"
            ] = replace_x_velocity
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_vy"
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

    def calculate_pitch_control_without_player(self):
        """
        Function description:
        This function calculates a pitch control surface after removing the player from the pitch.
        This can be used to attribute which spaces on the pitch are controlled by the specific player, rather than the
        space occupied by his/her team.

        Returns:
        edited_pitch_control: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control
                probability for the attacking team after removing the relevant player from the pitch. Surface for the
                defending team is simply 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
        """
        self._validate_inputs()

        # Replace player's datapoint nan's, so pitch control does not take into account
        # the player when computing its surface
        temp_tracking_home = self.tracking_home.copy()
        temp_tracking_away = self.tracking_away.copy()

        if self.team_player_to_analyze == "Home":
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_x"
            ] = np.nan
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_y"
            ] = np.nan
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_vx"
            ] = np.nan
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_vy"
            ] = np.nan
        elif self.team_player_to_analyze == "Away":
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_x"
            ] = np.nan
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_y"
            ] = np.nan
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_vx"
            ] = np.nan
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_vy"
            ] = np.nan

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

    def calculate_pitch_control_new_location(
        self,
        relative_x_change,
        relative_y_change,
        replace_velocity=False,
        replace_x_velocity=0,
        replace_y_velocity=0,
    ):

        """
        Function description:
        This function calculates a pitch control surface after moving the player to a new location on the pitch,
        and specifying a new velocity vector for the player.
        Ideally, this function would be used to identify/illustrate where players could be located/moving towards in
        order to maximize their team's pitch control

        Input parameters:
        :param float relative_x_change: The amount to change the x coordinate of the player by before calculating the
                new pitch control model. Measured in meters
        :param float relative_y_change: The amount to change the y coordinate of the player by before calculating the
                new pitch control model. Measured in meters.
        :param bool replace_velocity: Tells us whether to replace the player's velocity vector with a new one. If False,
            the player's velocity vector will remain the same. If True, the player's velocity will be placed with the
            values in the ``replace_x_velocity`` and  replace_y_velocity`` argument. Default is False.
        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with.
                Positive values will move the player toward's the home team's goal, while negative values will move the
                player towards the Away Team's goal. Measured in m/s. Defaults to 0.
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with.
                Positive values will move the player to the left side of the pitch, from the perspective of the away
                team, while negative values will move the player towards the right side of the pitch from the
                perspective of the away team. Measured in m/s. Defaults to 0.

        Returns:
        edited_pitch_control: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control
                probability for the attcking team with one player's velocity changed
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
        """
        self._validate_inputs()

        if replace_velocity & (replace_x_velocity == 0) & (replace_y_velocity == 0):
            warnings.warn(
                "You have not specified a new velocity vector for the player. All analysis will assume "
                "that the player is stationary in his/her new location"
            )

        # Replace datapoints with a new location and velocity vector
        temp_tracking_home = self.tracking_home.copy()
        temp_tracking_away = self.tracking_away.copy()

        if self.team_player_to_analyze == "Home":
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_x"
            ] += relative_x_change
            temp_tracking_home.at[
                self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_y"
            ] += relative_y_change
            if replace_velocity:
                temp_tracking_home.at[
                    self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_vx"
                ] = replace_x_velocity
                temp_tracking_home.at[
                    self.tracking_frame, "Home_" + str(self.player_to_analyze) + "_vy"
                ] = replace_y_velocity
        elif self.team_player_to_analyze == "Away":
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_x"
            ] += relative_x_change
            temp_tracking_away.at[
                self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_y"
            ] += relative_y_change
            if replace_velocity:
                temp_tracking_away.at[
                    self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_vx"
                ] = replace_x_velocity
                temp_tracking_away.at[
                    self.tracking_frame, "Away_" + str(self.player_to_analyze) + "_vy"
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
        self,
        replace_velocity=False,
        replace_x_velocity=0,
        replace_y_velocity=0,
        relative_x_change=0,
        relative_y_change=0,
        replace_function="movement",
    ):
        """
        Function description:
        This function computes the difference in pitch control surfaces between the actual event,and the event when
        one player's attributes (velocity, position, or presence) is altered. In this function, it is strongly
        recommended to use use the arguments that are relevant to the parent pitch control calculate function, depending
        on which effect we want to study.

        Input parameters:
        :param bool replace_velocity: This determines if we should replace our velocity vector with a new vector.
            Only used if ``replace_function = 'location'``.
        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with.
            Positive values will move the player toward's the home team's goal, while negative values will move the
            player towards the Away Team's goal. Measured in m/s. Not used if
            'replace_function' = 'presence'. Defaults to 0.
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with.
            Positive values will move the player to the left side of the pitch, from the perspective of the away team,
            while negative values will move the player towards the right side of the pitch from the perspective of the
            away team. Measured in m/s. Not used if ``replace_function = 'presence'`` Defaults to 0.
        :param float relative_x_change: Determines how much to move the player in the x direction,
            relative to his/her current location on the pitch. Positive values will move the player toward's the home
            team's goal, while negative values will move the player towards the Away Team's goal. Measured in meters.
            Not used if ``replace_function = 'presence'`` or ``replace_function == 'movement'``.
        :param float relative_y_change: Determines how much to move the player in the y direction,
            relative to his/her current location on the pitch. Positive values will move the player to the left side of
            the pitch, from the perspective of the away team, while negative values will move the player towards the
            right side of the pitch from the perspective of the away team. Measured in meters.
            Only used if ``replace_function='location'``.
        :param string replace_function: The type of function we are using to analyze the player.
            Must be either "movement", "presence" or "location". Defaults to "movement".
            The argument "movement" uses the function ``calculate_pitch_control_replaced_velocity`` to calculate a new
            pitch control surface, using your input for the player's new velocity. If no arguments are passed for
            velocity, it will calculate the difference in pitch control between the player's current velocity, and the
            pitch control if the player is stationary.
            The argument "presence" uses the function ``calculate_pitch_control_without_player`` to calculate a new
            pitch control surface, assuming the player is not on the pitch. The function then returns a difference in
            pitch control between the actual event, and the event if the player were not present.
            The argument "location" uses  the function ``calculate_pitch_control_new_location`` to calculate a new
            pitch control surface assuming the player is in a new location on the pitch with a new velocity vector.
            The function then returns a difference in pitch control between the actual event, and the event if the
            player were in his new location.

        Returns:
            pitch_control_difference: Difference in pitch control surfaces (dimen (n_grid_cells_x,n_grid_cells_y) )
            between the actual event, and the event with the player's movement, location and/or presence altered.
            Returns difference in pitch control with respect to the team that currently has possession of the ball
            The difference surface for the defending team is simply 1-PPCFa.
            xgrid: Positions of the pixels in the x-direction (field length)
            ygrid: Positions of the pixels in the y-direction (field width)
        """
        if replace_function == "movement":
            (
                edited_pitch_control,
                xgrid,
                ygrid,
            ) = self.calculate_pitch_control_replaced_velocity(
                replace_x_velocity=replace_x_velocity,
                replace_y_velocity=replace_y_velocity,
            )
        elif replace_function == "presence":
            (
                edited_pitch_control,
                xgrid,
                ygrid,
            ) = self.calculate_pitch_control_without_player()
        elif replace_function == "location":
            (
                edited_pitch_control,
                xgrid,
                ygrid,
            ) = self.calculate_pitch_control_new_location(
                replace_velocity=replace_velocity,
                replace_x_velocity=replace_x_velocity,
                replace_y_velocity=replace_y_velocity,
                relative_x_change=relative_x_change,
                relative_y_change=relative_y_change,
            )

        else:
            raise ValueError(
                "replace_function must be either 'movement', 'presence' or 'location'"
            )
        pitch_control_difference = self.event_pitch_control - edited_pitch_control
        return pitch_control_difference, xgrid, ygrid

    def calculate_space_created(
        self,
        replace_x_velocity=0,
        replace_y_velocity=0,
        relative_x_change=0,
        relative_y_change=0,
        replace_velocity=False,
        replace_function="movement",
    ):
        """
        Function description:
        This function calculates the total amount of space generated by a player, relative to the amount generated if
            the player's velocity, location or presence is changed. In this function, it is strongly recommended to use
            the arguments that are relevant to the parent pitch control calculation function, depending on which effect
            we want to study.

        Input parameters:
        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with.
            Positive values will move the player toward's the home team's goal, while negative values will move the
            player towards the Away Team's goal. Measured in m/s.
            ``Not used if replace_function='presence'``
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with.
            Positive values will move the player to the left side of the pitch, from the perspective of the away team,
            while negative values will move the player towards the right side of the pitch from the perspective of the
            away team. Measured in m/s. Not used if ``replace_function='presence``.
        :param float relative_x_change: Determines how much to move the player in the x direction,
            relative to his/her current location on the pitch. Positive values will move the player toward's the home
            team's goal, while negative values will move the player towards the Away Team's goal. Measured in meters.
            Not used if ``replace_function = 'presence'`` or ``replace_function == 'movement'``.
        :param float relative_y_change: Determines how much to move the player in the y direction,
            relative to his/her current location on the pitch. Positive values will move the player to the left side of
            the pitch, from the perspective of the away team, while negative values will move the player towards the
            right side of the pitch from the perspective of the away team. Measured in meters.
            Only used if ``replace_function='location'``.
        :param bool replace_velocity: This determines if we should replace our velocity vector with a new vector.
            Only used if ``replace_function = 'location'``.
        :param string replace_function: The type of function we are using to analyze the player.
            Must be either "movement", "presence" or "location". Defaults to "movement". For detailed description of
            this argument, please refer to the docstring in ``calculate_pitch_control_difference``

        Returns:
         A float representing amount of pitch gained (or lost) by the player's team, relative to the amount of space
            the player's team occupied before editing the player's velocity, location or movement.
            Positive values represent space lost by the player's team after editing the player's attributes,
             while negative values represent space gained after editing the player's attributes. Measured in m^2.
        """
        (
            pitch_control_difference,
            xgrid,
            ygrid,
        ) = self.calculate_pitch_control_difference(
            replace_x_velocity=replace_x_velocity,
            replace_y_velocity=replace_y_velocity,
            relative_x_change=relative_x_change,
            relative_y_change=relative_y_change,
            replace_function=replace_function,
            replace_velocity=replace_velocity,
        )

        pitch_control_change = self.calculate_total_space_on_pitch_team(
            pitch_control_result=pitch_control_difference, calculating_diff=True
        )
        pitch_control_change = pitch_control_change * (
            2 * (self.team_in_possession == self.team_player_to_analyze) - 1
        )
        return pitch_control_change

    def plot_pitch_control_difference(
        self,
        replace_x_velocity=0,
        replace_y_velocity=0,
        relative_x_change=0,
        relative_y_change=0,
        replace_velocity=False,
        replace_function="movement",
        cmap_list=[],
        alpha=0.7,
        alpha_pitch_control=0.5,
        team_colors=("r", "b"),
    ):
        """
        Function description:
        This function plots the difference between the actual event in the match, and the event if the player's velocity,
        presence, or movement is altered. In this function, it is strongly recommended to use the arguments that are
        relevant to the parent pitch control calculation function, depending on which effect we want to study.

        If we are removing the player from the pitch (by setting ``replace_function='presence'``), this function plots
        the space that the player was occupying for his team during the given event.
        If we are altering the player's velocity vector (by setting ``replace_function='movement'``), this function
        plots the space gained by the player's current movement in the color of the player's team, and plots the space
        lost by the player's current movement in the color of the opposite team.
        If we are moving the player's location on the pitch (by setting ``replace_function='location'``), we are
        plotting the space the player's team would gain by being in the new location in the player's team color, and
        the space that the player would concede by moving to the new location in the color of the opposition team. We
        also plot the player's proposed new location and velocity in green.

        Input Parameters:
        :param float replace_x_velocity: The x vector of the velocity we would like to replace our given player with.
            Positive values will move the player toward's the home team's goal, while negative values will move the
            player towards the Away Team's goal. Measured in m/s.
            ``Not used if replace_function='presence'``
        :param float replace_y_velocity: The y vector of the velocity we would like to replace our given player with.
            Positive values will move the player to the left side of the pitch, from the perspective of the away team,
            while negative values will move the player towards the right side of the pitch from the perspective of the
            away team. Measured in m/s. Not used if ``replace_function='presence``.
        :param float relative_x_change: Determines how much to move the player in the x direction,
            relative to his/her current location on the pitch. Positive values will move the player toward's the home
            team's goal, while negative values will move the player towards the Away Team's goal. Measured in meters.
            Not used if ``replace_function = 'presence'`` or ``replace_function == 'movement'``.
        :param float relative_y_change: Determines how much to move the player in the y direction,
            relative to his/her current location on the pitch. Positive values will move the player to the left side of
            the pitch, from the perspective of the away team, while negative values will move the player towards the
            right side of the pitch from the perspective of the away team. Measured in meters.
            Only used if ``replace_function='location'``.
        :param bool replace_velocity: This determines if we should replace our velocity vector with a new vector.
            Only used if ``replace_function = 'location'``.
        :param string replace_function: The type of function we are using to analyze the player.
            Must be either "movement", "presence" or "location". Defaults to "movement". For a detailed description of
            this argument, please refer to the docstring in ``calculate_pitch_control_difference``.

        :param list cmap_list: List of colors to use in the pitch control spaces for each team. Default is an empty list.
        :param float alpha: alpha (transparency) of player markers. Default is 0.7
        :param float alpha_pitch_control: alpha (transparency) of spaces heatmap. Default is 0.5
        :param tuple team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)


        Returns:
            This function technically does not return anything, but does produce a matplotlib plot.
        """
        (
            pitch_control_difference,
            xgrid,
            ygrid,
        ) = self.calculate_pitch_control_difference(
            replace_x_velocity=replace_x_velocity,
            replace_y_velocity=replace_y_velocity,
            relative_x_change=relative_x_change,
            relative_y_change=relative_y_change,
            replace_function=replace_function,
            replace_velocity=replace_velocity,
        )

        if replace_function == "presence":
            mviz.plot_pitchcontrol_for_event(
                event_id=self.event_id,
                events=self.events,
                tracking_home=self.tracking_home,
                tracking_away=self.tracking_away,
                PPCF=pitch_control_difference,
                annotate=True,
                xgrid=xgrid,
                ygrid=ygrid,
                plotting_presence=True,
                team_to_plot=self.team_player_to_analyze,
                alpha=alpha,
                alpha_pitch_control=alpha_pitch_control,
                cmap_list=cmap_list,
                team_colors=team_colors,
            )
        elif replace_function == "movement":

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
                alpha=alpha,
                alpha_pitch_control=alpha_pitch_control,
                cmap_list=cmap_list,
                team_colors=team_colors,
            )
        elif replace_function == "location":
            if self.team_player_to_analyze == "Home":
                x_coordinate = (
                    self.tracking_home.loc[self.tracking_frame][
                        "Home_" + str(self.player_to_analyze) + "_x"
                    ]
                    + relative_x_change
                )
                y_coordinate = (
                    self.tracking_home.loc[self.tracking_frame][
                        "Home_" + str(self.player_to_analyze) + "_y"
                    ]
                    + relative_y_change
                )
                if replace_velocity == False:
                    replace_x_velocity = self.tracking_home.loc[self.tracking_frame][
                        "Home_" + str(self.player_to_analyze) + "_vx"
                    ]
                    replace_y_velocity = self.tracking_home.loc[self.tracking_frame][
                        "Home_" + str(self.player_to_analyze) + "_vy"
                    ]
            else:
                x_coordinate = (
                    self.tracking_away.loc[self.tracking_frame][
                        "Away_" + str(self.player_to_analyze) + "_x"
                    ]
                    + relative_x_change
                )
                y_coordinate = (
                    self.tracking_away.loc[self.tracking_frame][
                        "Away_" + str(self.player_to_analyze) + "_y"
                    ]
                    + relative_y_change
                )
                if replace_velocity == False:
                    replace_x_velocity = self.tracking_away.loc[self.tracking_frame][
                        "Away_" + str(self.player_to_analyze) + "_vx"
                    ]
                    replace_y_velocity = self.tracking_away.loc[self.tracking_frame][
                        "Away_" + str(self.player_to_analyze) + "_vy"
                    ]

            mviz.plot_pitchcontrol_for_event(
                event_id=self.event_id,
                events=self.events,
                tracking_home=self.tracking_home,
                tracking_away=self.tracking_away,
                PPCF=pitch_control_difference,
                annotate=True,
                xgrid=xgrid,
                ygrid=ygrid,
                plotting_new_location=True,
                plotting_difference=True,
                player_id=self.player_to_analyze,
                player_x_coordinate=x_coordinate,
                player_y_coordinate=y_coordinate,
                player_x_velocity=replace_x_velocity,
                player_y_velocity=replace_y_velocity,
                alpha=alpha,
                alpha_pitch_control=alpha_pitch_control,
                cmap_list=cmap_list,
                team_colors=team_colors,
            )
        if replace_function == "movement":
            plt.title(
                "Space created by "
                + str(self.team_player_to_analyze)
                + " Player "
                + str(self.player_to_analyze)
                + " during event "
                + str(self.event_id),
                fontdict={"fontsize": 22},
            )
        elif replace_function == "presence":
            plt.title(
                "Space occupied by "
                + str(self.team_player_to_analyze)
                + " Player "
                + str(self.player_to_analyze)
                + " during event "
                + str(self.event_id),
                fontdict={"fontsize": 22},
            )
        elif replace_function == "location":
            plt.title(
                "Difference In Pitch Control After Moving "
                + str(self.team_player_to_analyze)
                + " Player "
                + str(self.player_to_analyze),
                fontdict={"fontsize": 22},
            )

    def partial_space_creation(self, params):
        """
        Function Description:
            This function is used to help determine the optimal location on the pitch, by calculating the total space
            controlled by a team after altering the location and/or velocity vector of a given player.

        Input Parameters:
        The parameters of this function are stored in a dictionary (``params``) to be compatible with hyperopt's
        optimization functions. Below, we describe each key of the dictionary, and what their values represent in the
        function.

        :param float params['x_change']: The amount to change the x location of the player. Measured in meters.
        :param float params['y_change']: The amount to change the y location of the player. Measured in meters.
        :param float params['velocity']: The new velocity of the player, measured in meters per second.
        :param float params['angle']: New angle of the player, relative to the positive x direction (towards the home
            team's goal) and the positive y direction (towards the left side of the pitch from the perspective of the
            away team). Measured in radians

        Returns:
        :return: The total space on the pitch controlled by the player's team, after altering the location and/or
            velocity vector of the given player
        """
        x_change = params["x_change"]
        y_change = params["y_change"]
        velocity = params["velocity"]
        angle = params["angle"]
        x_velocity = velocity * np.sin(angle)
        y_velocity = velocity * np.cos(angle)
        new_pitch_control, xgrid, ygrid = self.calculate_pitch_control_new_location(
            relative_x_change=x_change,
            relative_y_change=y_change,
            replace_velocity=True,
            replace_x_velocity=x_velocity,
            replace_y_velocity=y_velocity,
        )
        space_creation = self.calculate_total_space_on_pitch_team(
            new_pitch_control, calculating_diff=False
        )
        print(params, space_creation)
        return -1 * space_creation

    def get_optimal_location_on_pitch(
        self, size_of_grid=20, location_trials=50, velocity_trials=0, max_velocity=5
    ):
        """
        Function description:
        This function attempts to find an "optimal location" on the pitch for a specific player, dependent on the
            locations and velocity vectors of the 21 players. Uses the package ``hyperopt`` to try to maximize the
            pitch control of the team, given the player is within a certain square box of his or her location. In
            addition, we can estimate the optimal velocity vector, subject to a user defined max velocity. The general
            flow is we optimize the location first (assuming 0 velocity), then the velocity. The user can choose to
            optimize location, velocity or both using the location_trials and velocity_trials parameters.

        Input parameters:
        :param float/int size_of_grid: A square box drawn around the player's current location to test a potential
            "optimal location". Measured in meters. Defaults to 20 (10 meters each direction from the player).
        :param int location_trials: The number of trials to try to determine the optimal location. More trials means
            it is more likely you will find the optimal location, but the runtime is longer (scales linearly).
            If set to 0, the player's location will remain constant, and we will only test the velocity vector.
            Defaults to 50.
        :param int velocity_trials: The number of trials to try to determine the optimal velocity vector. More trials
            means it is more likely you will find the optimal value, but the runtime is longer (scales linearly). If
            set to 0, we will skip this portion of the code, and the function will only return the optimal location,
            with a velocity of 0.
        :param float/int max_velocity: The maximum allowed velocity of a player when trying to compute his/her optimal
            velocity vector. Measured in meters per second Defaults to 5.

        Returns:
        :return: A plot of the player's estimated optimal location and/or velocity vector on the pitch, with shadings
            for space gained and conceded by adjusting the location and velocity/vector
        """
        if (location_trials == 0) * (velocity_trials == 0):
            raise ValueError(
                "One of location_trials or velocity_trials must be greater than 0"
            )
        if size_of_grid <= 0:
            raise ValueError("size_of_grid must be greater than 0")
        offside_position = self._determine_offside_position()
        max_y_coord = self.field_dimens[1] / 2
        min_y_coord = -1 * max_y_coord
        if self.team_player_to_analyze == "Home":
            player_x_coordinate = self.tracking_home.loc[self.tracking_frame][
                "Home_" + str(self.player_to_analyze) + "_x"
            ]
            player_y_coordinate = self.tracking_home.loc[self.tracking_frame][
                "Home_" + str(self.player_to_analyze) + "_y"
            ]
            min_x_coord = offside_position
            max_x_coord = self.field_dimens[0] / 2
        elif self.team_player_to_analyze == "Away":
            player_x_coordinate = self.tracking_away.loc[self.tracking_frame][
                "Away_" + str(self.player_to_analyze) + "_x"
            ]
            player_y_coordinate = self.tracking_away.loc[self.tracking_frame][
                "Away_" + str(self.player_to_analyze) + "_y"
            ]
            min_x_coord = -1 * self.field_dimens[0] / 2
            max_x_coord = offside_position
        else:
            raise ValueError("team_player_to_analyze must be either 'Home' or 'Away'")

        if location_trials > 0:
            location_space = {
                "x_change": hp.uniform(
                    "x_change",
                    max(min_x_coord - player_x_coordinate, -1 * size_of_grid / 2),
                    min(max_x_coord - player_x_coordinate, size_of_grid / 2),
                ),
                "y_change": hp.uniform(
                    "y_change",
                    max(min_y_coord - player_y_coordinate, -1 * size_of_grid / 2),
                    min(max_y_coord - player_y_coordinate, size_of_grid / 2),
                ),
                "velocity": hp.uniform("x_velocity", -0.001, 0.001),
                "angle": hp.uniform("y_velocity", -0.001, 0.001),
            }

            location_optimization = fmin(
                fn=self.partial_space_creation,
                space=location_space,
                algo=tpe.suggest,
                max_evals=location_trials,
                trials=Trials(),
            )
        elif location_trials == 0:
            location_optimization = {
                "x_change": 0,
                "y_change": 0,
                "x_velocity": 0,
                "y_velocity": 0,
            }
        else:
            raise ValueError("location_trials must be greater than or equal to 0")

        if velocity_trials > 0:
            velocity_space = {
                "x_change": hp.uniform(
                    "x_change",
                    location_optimization["x_change"] - 0.01,
                    location_optimization["x_change"] + 0.01,
                ),
                "y_change": hp.uniform(
                    "y_change",
                    location_optimization["y_change"] - 0.01,
                    location_optimization["y_change"] + 0.01,
                ),
                "velocity": hp.uniform(
                    "velocity",
                    -1 * np.sqrt(max_velocity ** 2 / 2),
                    np.sqrt(max_velocity ** 2 / 2),
                ),
                "angle": hp.uniform("angle", 0, 2 * math.pi),
            }
            velocity_optimization = fmin(
                fn=self.partial_space_creation,
                space=velocity_space,
                algo=tpe.suggest,
                max_evals=velocity_trials,
                trials=Trials(),
            )
        elif velocity_trials == 0:
            velocity_optimization = {
                "x_change": location_optimization["x_change"],
                "y_change": location_optimization["y_change"],
                "x_velocity": 0,
                "y_velocity": 0,
            }
        else:
            raise ValueError("velocity_trials must be greater than or equal to 0")

        # return velocity_optimization
        self.plot_pitch_control_difference(
            replace_function="location",
            relative_x_change=velocity_optimization["x_change"],
            relative_y_change=velocity_optimization["y_change"],
            replace_x_velocity=velocity_optimization["velocity"]
            * np.sin(velocity_optimization["angle"]),
            replace_y_velocity=velocity_optimization["velocity"]
            * np.cos(velocity_optimization["angle"]),
            replace_velocity=True,
        )
        if location_trials == 0:
            plt.title(
                "Optimal velocity vector of "
                + self.team_player_to_analyze
                + " Player "
                + str(self.player_to_analyze)
                + " during Event "
                + str(self.event_id),
                fontdict={"fontsize": 22},
            )
        else:
            plt.title(
                "Optimal location of "
                + self.team_player_to_analyze
                + " Player "
                + str(self.player_to_analyze)
                + " during Event "
                + str(self.event_id),
                fontdict={"fontsize": 22},
            )
        plt.show()
        return

    def _determine_offside_position(self):
        """
        Function Description:
            This function determines the x value that would make a player offside, to aid us in determining the
            "optimal location" for a player. Determined by taking the second most extreme value of the opposing team's
            tracking frame for the given event.

        :return: float The x value that would make a player offside.
        """
        x_coordinates = []
        if self.team_player_to_analyze == "Home":
            players_on_pitch = self._get_players_on_pitch(team="Away")
            tracking_frame = self.tracking_away.loc[self.tracking_frame]
            for player in players_on_pitch:
                player_x_position = tracking_frame["Away_" + str(player) + "_x"]
                x_coordinates.append(player_x_position)
            second_lowest_x = sorted(x_coordinates, reverse=False)[1]
            return second_lowest_x
        elif self.team_player_to_analyze == "Away":
            players_on_pitch = self._get_players_on_pitch(team="Home")
            tracking_frame = self.tracking_home.loc[self.tracking_frame]
            for player in players_on_pitch:
                player_x_position = tracking_frame["Home_" + str(player) + "_x"]
                x_coordinates.append(player_x_position)
            second_highest_x = sorted(x_coordinates, reverse=True)[1]
            return second_highest_x

    def _get_players_on_pitch(self, team):
        pass_frame = self.events.loc[self.event_id]["Start Frame"]
        players_on_pitch = []
        if team == "Home":
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

        if str(self.player_to_analyze) not in self._get_players_on_pitch(
            team=self.team_player_to_analyze
        ):
            raise ValueError(
                "player_to_analyze is either not on the correct team, or was not on the pitch at the time of the event"
            )
