# FoT-Player-Pitch-Control-Impact

In this repo, we take a look at the challenge put out by Laurie Shaw (@EightyFivePoint) in his 3rd tutorial in the Friends Of Tracking Series: https://www.youtube.com/watch?v=5X1cSehLg6s

The challenge posed at the end of the video was to use the pitch control model presented in the video to calculate how much space was created (or territory captured) by an off the ball run

In the file PlayerPitchControlAnalysis.py, we build out some tools to answer this question, and visualize the space that is being created.

We build out the following tools, given a player/teamID and an event from the corresponding Metrica events dataframe:

1. Calculate the amount of space occupied on the pitch (per EightyFivePoint's pitch control model) in meters squared
2. Calculate the difference in total space occupied by the player's team with his current movement vs theoretical movement
3. Compute the pitch control surface if a specific player had a different velocity vector
4. Compute the difference in the pitch control surface between a player's actual velocity vector, and a theorized one
5. The ability to plot the space created (and lost) by a player's actual movement, relative to a theoretical velocity vector


Data courtesy of Bruno Dagnino (Metrica Sports).

All helper functions outside of this file courtesy of Laurie Shaw, and can be found in the following Github repository: https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking.

If you have any questions about this code, please feel free to DM me on Twitter (@andrew_puopolo) or email me (puopolo4@gmail.com) 
