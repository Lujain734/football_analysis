# team_assigner.py
import numpy as np
from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        """Initializes the TeamAssigner."""
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None # Will store the model used for 2-team clustering

    def get_clustering_model(self, image):
        """
        Creates a KMeans model configured to find the single dominant color
        in a player crop. Does NOT fit the model here.
        """
        # Reshape the image to 2D array (list of pixels [R, G, B])
        image_2d = image.reshape(-1, 3)

        # --- Check if the reshaped array is empty ---
        if image_2d.shape[0] == 0:
            print("⚠️ Warning: get_clustering_model received image leading to 0 samples.")
            return None # Indicate failure to create a model

        # Create K-means model with 1 cluster to find the dominant color.
        # n_init='auto' is generally recommended over a fixed number like 1 or 10.
        kmeans = KMeans(n_clusters=1, init="k-means++", n_init='auto', random_state=0)

        return kmeans # Return the *unfitted* model

    def get_player_color(self, frame, player_bbox):
        """
        Extracts the dominant color from the top half of a player's bounding box.
        Includes robust checks for invalid bounding boxes and empty image crops.
        """
        x1, y1, x2, y2 = map(int, player_bbox)

        # --- Coordinate clamping ---
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_frame - 1, x2) # Use w_frame-1 and h_frame-1 for valid slicing
        y2 = min(h_frame - 1, y2)

        # --- Check for invalid/zero-area box after clamping ---
        if x1 >= x2 or y1 >= y2:
            # print(f"DEBUG: Invalid bbox after clamp: [{x1},{y1},{x2},{y2}]")
            return (0, 0, 0) # Return default black for invalid box

        player_image = frame[y1:y2, x1:x2]

        # --- Check if the initial crop is empty ---
        if player_image.size == 0:
            # print(f"DEBUG: player_image size 0 for bbox [{x1},{y1},{x2},{y2}]")
            return (0, 0, 0)

        h, w = player_image.shape[:2]
        top_half_pixels = int(h / 2)

        # --- Check if the crop is too small for a top half ---
        if top_half_pixels < 1:
            # print(f"DEBUG: top_half_pixels < 1 (h={h}) for bbox [{x1},{y1},{x2},{y2}]")
            # If height is 0 or 1, can't get a meaningful top half for clustering
            return (0, 0, 0)

        top_half_image = player_image[:top_half_pixels, :]

        # --- Check if the top_half_image itself is empty ---
        if top_half_image.size == 0:
            # print(f"DEBUG: top_half_image size 0 for bbox [{x1},{y1},{x2},{y2}]")
            return (0, 0, 0)

        # Reshape for fitting (list of pixels)
        image_2d = top_half_image.reshape(-1, 3)

        # --- Final check before fitting ---
        if image_2d.shape[0] == 0:
            # print(f"DEBUG: image_2d shape 0 for bbox [{x1},{y1},{x2},{y2}]")
            return (0, 0, 0)

        try:
            # Get the *unfitted* model (n_clusters=1)
            kmeans = self.get_clustering_model(top_half_image)
            if kmeans is None: # Handle case where get_clustering_model found 0 samples
                 # print(f"DEBUG: kmeans model is None for bbox [{x1},{y1},{x2},{y2}]")
                 return (0, 0, 0)

            # Fit the model to the valid pixel data
            kmeans.fit(image_2d)

            # Get the dominant color (the center of the single cluster)
            dominant_color = kmeans.cluster_centers_[0]
            return tuple(map(int, dominant_color)) # Return as (R, G, B) tuple

        except ValueError as e:
            # Catch potential errors during KMeans fitting
            print(f"⚠️ KMeans error during player color fit: {e}")
            print(f"   Image shape: {top_half_image.shape}, BBox: [{x1},{y1},{x2},{y2}]")
            return (0, 0, 0) # Return default color on error

    def assign_team_color(self, frame, player_detections):
        """
        Determines the two primary team colors based on dominant player colors
        from the first frame (or specified frame).
        """
        player_colors = []
        valid_colors_count = 0
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            # Only use non-default (non-black) colors for team assignment
            if player_color != (0, 0, 0):
                player_colors.append(player_color)
                valid_colors_count += 1

        # Check if we have enough valid colors to cluster into 2 teams
        if valid_colors_count < 2:
             print("⚠️ Warning: Could not find enough distinct player colors (<2) to determine 2 teams. Assigning default colors.")
             # Assign arbitrary distinct colors as fallback
             self.team_colors[1] = np.array([255, 0, 0]) # Red Team 1
             self.team_colors[2] = np.array([0, 0, 255]) # Blue Team 2
             self.kmeans = None # Indicate that kmeans wasn't properly fitted for teams
             return

        # Use n_clusters=2 for separating the two teams based on player colors
        try:
             # Use numpy array for fitting
             player_colors_np = np.array(player_colors)
             # n_init='auto' is generally better
             kmeans_teams = KMeans(n_clusters=2, init="k-means++", n_init='auto', random_state=0)
             kmeans_teams.fit(player_colors_np)

             self.kmeans = kmeans_teams # Store the fitted model for team prediction
             self.team_colors[1] = self.kmeans.cluster_centers_[0] # Assign color for Team 1
             self.team_colors[2] = self.kmeans.cluster_centers_[1] # Assign color for Team 2
             print(f"✅ Assigned team colors: Team 1 ~{self.team_colors[1]}, Team 2 ~{self.team_colors[2]}")

        except ValueError as e:
             # This might happen if all valid colors are identical, etc.
             print(f"⚠️ KMeans error during team assignment fit: {e}")
             print(f"   Number of valid player colors: {len(player_colors)}")
             # Assign arbitrary distinct colors as fallback
             self.team_colors[1] = np.array([255, 0, 0]) # Red Team 1
             self.team_colors[2] = np.array([0, 0, 255]) # Blue Team 2
             self.kmeans = None

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predicts the team (1 or 2) for a player based on their dominant color,
        using the previously fitted 2-cluster team model.
        Caches results by player_id.
        """
        # Return cached team ID if already determined
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        # Handle cases where color couldn't be determined or team model failed
        if player_color == (0, 0, 0):
             print(f"⚠️ Could not determine color for player {player_id}, defaulting team.")
             team_id = 1 # Default to Team 1 (arbitrary choice)
        elif self.kmeans is None:
             # Kmeans failed during assign_team_color, default assignment
             print(f"⚠️ Team color model (kmeans) not available, defaulting team for player {player_id}.")
             team_id = 1 # Default to Team 1
        else:
            # Predict team using the fitted 2-cluster kmeans model
            try:
                # Need to reshape the single color for prediction
                player_color_np = np.array(player_color).reshape(1, -1)
                team_id_index = self.kmeans.predict(player_color_np)[0] # Returns 0 or 1
                team_id = team_id_index + 1 # Convert index (0 or 1) to team_id (1 or 2)
            except Exception as e:
                print(f"⚠️ Error predicting team for player {player_id} with color {player_color}: {e}")
                team_id = 1 # Default on prediction error

        # --- Special override (Use with caution or remove if not needed) ---
        # Example: if player_id == 91:
        #     print(f"DEBUG: Overriding team for player {player_id} to 1")
        #     team_id = 1
        # --- End Special override ---

        # Cache the result
        self.player_team_dict[player_id] = team_id
        return team_id
