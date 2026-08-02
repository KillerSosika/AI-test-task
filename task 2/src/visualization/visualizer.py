import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class Visualizer:
    @staticmethod
    def draw_matches(img_before: np.ndarray, img_after: np.ndarray, mkpts0: np.ndarray, mkpts1: np.ndarray, mconf=None, title="Matched Keypoints"):
        """
        Draws matches. Filters out lines for low-confidence (red) points to reduce visual noise.
        """
        # Correct normalization for 16-bit satellite data
        def prep(img):
            img_f = img.astype(np.float32)
            if img_f.max() > 255.0:
                img_f = np.clip(img_f / 3000.0, 0, 1)
            elif img_f.max() > 1.0:
                img_f = np.clip(img_f / 255.0, 0, 1)
            return (img_f * 255).astype(np.uint8)

        img_b = prep(img_before)
        img_a = prep(img_after)

        # Stitch images horizontally
        h1, w1 = img_b.shape[:2]
        h2, w2 = img_a.shape[:2]
        combined = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
        combined[:h1, :w1] = img_b
        combined[:h2, w1:w1+w2] = img_a

        plt.figure(figsize=(18, 9))
        plt.title(title, fontsize=18, fontweight='bold', pad=15)
        plt.imshow(combined, cmap='gray')

        # Set colors and normalization (if confidence is available)
        if mconf is not None and len(mconf) > 0:
            c_min, c_max = mconf.min(), mconf.max()
            mconf_norm = (mconf - c_min) / (c_max - c_min) if c_max > c_min else np.zeros_like(mconf)
            cmap = mpl.colormaps.get_cmap('RdYlGn') 
            colors = cmap(mconf_norm)
        else:
            # If confidence is missing (SIFT, ORB), draw everything in blue
            colors = np.full((len(mkpts0), 4), (0.0, 0.5, 1.0, 0.7))
            mconf_norm = np.ones(len(mkpts0)) # Artificial normalization = 1.0 so all lines are shown

        # Draw each line individually
        for i in range(len(mkpts0)):
            x0, y0 = mkpts0[i]
            x1, y1 = mkpts1[i]
            x1 += w1 # Shift for the second image
            
            # --- VISUAL FILTER ---
            # Draw the connection line only for green and yellow points (> 0.5)
            if mconf_norm[i] >= 0.5:
                plt.plot([x0, x1], [y0, y1], color=colors[i], linewidth=1.0, alpha=0.8)
            
            # Render the points always, but make red ones less visible (alpha=0.2)
            alpha_pt = 0.8 if mconf_norm[i] >= 0.5 else 0.2
            plt.scatter([x0, x1], [y0, y1], color=colors[i], s=5, alpha=alpha_pt)

        # Add a colorbar (confidence scale) on the side
        if mconf is not None and len(mconf) > 0:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=c_min, vmax=c_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.02)
            cbar.set_label('Confidence Score', rotation=270, labelpad=20, fontsize=12)

        plt.axis('off')
        plt.tight_layout()
        plt.show()