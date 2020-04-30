# FoT-Player-Pitch-Control-Impact

In this repo, we investigate at the challenge put out by Laurie Shaw (@EightyFivePoint) in his 3rd tutorial in the Friends Of Tracking Series: https://www.youtube.com/watch?v=5X1cSehLg6s

The challenge posed at the end of the video was to use the pitch control model presented in the video to calculate how much space was created (or territory captured) by an off the ball run.

In the file PlayerPitchControlAnalysis.py, we build out some tools to analyze the impact a player's location and movement has on the pitch control during a given event of a match.

We build out tools to answer the following questions given a player/teamID and an event from the corresponding Metrica events dataframe:

1.) How can we quantify how much space a team occupies on the pitch?

2.) How can we quantify and visualize the effect of an off the ball run using pitch control?

3.) How can we quantify and visualize the space occupied by a player during a given event?

4.) How would the pitch control change if a player were in a different location on the pitch? Is the player in the optimal position given the locations and velocities of the other 21 players on the pitch?

Examples of how these tools can be applied are found in ``example_player_analysis.py``, where we generate the relevant plots and space creation metrics for one player on each team.

In addition, there are functions added to ``metrica_viz.py`` to help support the new player specific faults

Data courtesy of Bruno Dagnino (Metrica Sports).

All helper functions outside of this file courtesy of Laurie Shaw, and can be found in the following Github repository: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.

I am incredibly open to an expansion of this codebase, and would love for collaboration if anyone is interested. If you have ideas for optimizing, expanding or cleaning up this code, please feel free to submit a pull request, submit an issue or contact me on Twitter (@andrew_puopolo) or email (puopolo4@gmail.com) 
