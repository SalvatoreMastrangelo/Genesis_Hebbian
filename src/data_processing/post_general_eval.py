import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re


class URDFHistogramPlotter:

    def __init__(self, csv_path, save_path="drone_plot.png"):
        self.df = pd.read_csv(csv_path)
        self.save_path = save_path

    # ------------------------------------------------------------
    #   Normalizzazioni e trasformazioni singola riga
    # ------------------------------------------------------------
    def _process_row(self, row):
        """Restituisce tutte le metriche normalizzate per un drone."""

        # Cost of Transport = -negE
        cot_base = -row["f_negE_baseline1"]
        cot_tr = -row[["f_negE_trained1", "f_negE_trained2", "f_negE_trained3"]].values

        # COT: se == 100 → forzalo a 0
        cot_base = 0 if cot_base == 100 else cot_base
        cot_tr = np.array([0 if v == 100 else v for v in cot_tr])

        metrics = {
            "baseline": {
                "speed":  row["f_speed_baseline1"] / 24,
                "negE":   cot_base / 1,
                "prog":   row["f_prog_baseline1"] / 1000,
                "reward": row["reward_ep_mean_baseline1"] / 1000,
            },
            "trained": {
                "speed": np.array([
                    row["f_speed_trained1"] / 24,
                    row["f_speed_trained2"] / 24,
                    row["f_speed_trained3"] / 24,
                ]),
                "negE": cot_tr / 1,
                "prog": np.array([
                    row["f_prog_trained1"] / 1000,
                    row["f_prog_trained2"] / 1000,
                    row["f_prog_trained3"] / 1000,
                ]),
                "reward": np.array([
                    row["reward_ep_mean_trained1"] / 1000,
                    row["reward_ep_mean_trained2"] / 1000,
                    row["reward_ep_mean_trained3"] / 1000,
                ]),
            },
        }

        return metrics

    # ------------------------------------------------------------
    #   PLOT COMPLETO
    # ------------------------------------------------------------
    def plot(self):

        metric_order = ["speed", "negE", "prog", "reward"]
        metric_labels = {
            "speed": "Speed",
            "negE": "Cost of Transport",
            "prog": "Progress",
            "reward": "Reward",
        }
        legend_labels = [
            "Speed (/24)",
            "Cost of Transport (/1)",
            "Progress (/1000)",
            "Reward (/1000)",
        ]

        colors = {
            "speed":  "#1f77b4",
            "negE":   "#ff7f0e",
            "prog":   "#2ca02c",
            "reward": "#d62728",
        }

        # Sfondi GP/SP trasparenti
        policy_bg_colors = {
            "gp": (0.2, 0.4, 1.0, 0.08),   # azzurro leggero
            "sp": (1.0, 0.3, 0.3, 0.08),   # rosso leggero
        }

        n_drones = len(self.df)
        fig, ax = plt.subplots(figsize=(20, 7))

        # ------------------------------------------------------------
        # Linea tratteggiata riferita al Progress (300/1000)
        # ------------------------------------------------------------
        progress_y = 300 / 1000
        ax.axhline(progress_y,
                   color=colors["prog"],
                   linestyle="--",
                   linewidth=1.5,
                   alpha=0.8)

        # ------------------------------------------------------------
        # Geometria dei blocchi
        # ------------------------------------------------------------
        group_spacing = 2.5      # distanza tra droni
        gp_sp_gap = 1.1          # distanza tra GP e SP
        bar_spacing = 0.22       # distanza metriche
        bar_width = 0.12

        xticks = []
        xticklabels = []

        # ============================================================
        #   CALCOLO DELLE MEDIE ESATTAMENTE COME NEL PLOT
        # ============================================================

        general_bar_values = {m: [] for m in metric_order}
        special_bar_values = {m: [] for m in metric_order}
        per_drone_GP = []
        per_drone_SP = []

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp = metrics["baseline"]
            sp = metrics["trained"]

            drone_gp_vals = {}
            drone_sp_vals = {}

            for m in metric_order:

                # === GENERAL (baseline) ===
                gp_val = gp[m]
                if gp_val > 0:      # solo barre > 0
                    general_bar_values[m].append(gp_val)
                    drone_gp_vals[m] = gp_val
                else:
                    drone_gp_vals[m] = 0

                # === SPECIALIZED (trained) ===
                vals = sp[m]

                if m in ("speed", "negE"):
                    valid_mask = vals > 0
                else:
                    valid_mask = np.ones_like(vals, dtype=bool)

                valid_vals = vals[valid_mask]

                if len(valid_vals) > 0:
                    sp_mean = valid_vals.mean()
                    special_bar_values[m].append(sp_mean)
                    drone_sp_vals[m] = sp_mean
                else:
                    drone_sp_vals[m] = 0

            per_drone_GP.append(drone_gp_vals)
            per_drone_SP.append(drone_sp_vals)

        # === MEDIE FINALI DELLE BARRE ===
        final_general = {m: (np.mean(general_bar_values[m]) if len(general_bar_values[m]) else 0)
                        for m in metric_order}
        final_special = {m: (np.mean(special_bar_values[m]) if len(special_bar_values[m]) else 0)
                        for m in metric_order}

        print("Final General Policy Means:")
        for m in metric_order:
            print(f"  {m}: {final_general[m]:.4f}")

        print("Final Specialized Policy Means:")
        for m in metric_order:
            print(f"  {m}: {final_special[m]:.4f}")


        # ============================================================
        #   DELTA + STD (coerenti al plot)
        # ============================================================

        print("\n=== DELTA ASSOLUTO TRA LE MEDIE (COERENTE AL PLOT) ===")

        for m in metric_order:

            gp_mean = final_general[m]
            sp_mean = final_special[m]

            delta_global = gp_mean - sp_mean

            # Delta per drone solo se entrambi validi
            delta_per_drone = []

            for gp_d, sp_d in zip(per_drone_GP, per_drone_SP):
                if gp_d[m] > 0 and sp_d[m] > 0:
                    delta_per_drone.append(gp_d[m] - sp_d[m])

            if len(delta_per_drone) > 1:
                std_delta = np.std(delta_per_drone)
            else:
                std_delta = 0.0

            print(f"{m.upper():>8}: delta = {delta_global:+.4f}, std = {std_delta:.4f}")

        print("=====================================================\n")



        # ------------------------------------------------------------
        # LOOP DRONI (ogni riga = un drone)
        # ------------------------------------------------------------
        for idx, (_, row) in enumerate(self.df.iterrows(), start=1):

            metrics = self._process_row(row)

            # centro del drone
            base_x = idx * group_spacing

            # centri dei due gruppi
            gp_center = base_x - gp_sp_gap / 2
            sp_center = base_x + gp_sp_gap / 2

            # xtick principale
            xticks.append(base_x)
            label = f"DRONE {idx}"
            if idx == 15:
                label = "BIX3"

            xticklabels.append(label)


            # --------------------------------------------------------
            # Sfondi semitrasparenti GP e SP
            # --------------------------------------------------------
            gp_left  = gp_center - 2 * bar_spacing
            gp_right = gp_center + 2 * bar_spacing
            ax.axvspan(gp_left, gp_right,
                       facecolor=policy_bg_colors["gp"],
                       edgecolor=None)

            sp_left  = sp_center - 2 * bar_spacing
            sp_right = sp_center + 2 * bar_spacing
            ax.axvspan(sp_left, sp_right,
                       facecolor=policy_bg_colors["sp"],
                       edgecolor=None)

            # --------------------------------------------------------
            #   General Policy (baseline)
            # --------------------------------------------------------
            for mi, metric in enumerate(metric_order):
                x = gp_center + (mi - 1.5) * bar_spacing
                val = metrics["baseline"][metric]

                if val == 0:
                    ax.text(x, 0.01, "X",
                            ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(x, val, width=bar_width,
                           color=colors[metric], alpha=0.9)

            # --------------------------------------------------------
            #   Specialized Policy (trained)
            # --------------------------------------------------------
            for mi, metric in enumerate(metric_order):
                x = sp_center + (mi - 1.5) * bar_spacing
                vals = metrics["trained"][metric]

                # ------------------------------
                # FILTRAGGIO VALORI INVALIDI
                # ------------------------------
                if metric == "speed":
                    valid_mask = vals > 0
                elif metric == "negE":   # Cost of Transport
                    valid_mask = vals > 0
                else:
                    valid_mask = np.ones_like(vals, dtype=bool)

                valid_vals = vals[valid_mask]

                # ------------------------------
                # MEDIA SOLO DEI VALIDI
                # ------------------------------
                if len(valid_vals) == 0:
                    mean_val = 0
                else:
                    mean_val = valid_vals.mean()

                # ------------------------------
                # DISEGNO BARRA MEDIA
                # ------------------------------
                if mean_val == 0:
                    ax.text(x, 0.01, "X",
                            ha="center", va="bottom", fontsize=9)
                else:
                    ax.bar(x, mean_val, width=bar_width,
                        color=colors[metric], alpha=0.9)

                # ------------------------------
                # DISEGNO SEMPRE TUTTI I PALLINI
                # ------------------------------
                xs = np.full(len(vals), x)
                ax.scatter(xs, vals,
                        color=colors[metric],
                        edgecolor="black",
                        alpha=0.5)


            # --------------------------------------------------------
            #   Etichette "General Policy" e "Specialized Policy"
            # --------------------------------------------------------
            ax.text(gp_center, -0.08, "General",
                    ha="center", va="top", fontsize=10,
                    transform=ax.get_xaxis_transform())

            ax.text(sp_center, -0.08, "Specialized",
                    ha="center", va="top", fontsize=10,
                    transform=ax.get_xaxis_transform())

        # ============================================================
        # COLONNA MEDIA FINALE
        # ============================================================

        mean_base_x = (n_drones + 1) * group_spacing

        gp_center = mean_base_x - gp_sp_gap / 2
        sp_center = mean_base_x + gp_sp_gap / 2

        xticks.append(mean_base_x)
        xticklabels.append("MEAN")

        # Sfondo GP/SP
        gp_left  = gp_center - 2 * bar_spacing
        gp_right = gp_center + 2 * bar_spacing
        ax.axvspan(gp_left, gp_right, facecolor=policy_bg_colors["gp"], edgecolor=None)

        sp_left  = sp_center - 2 * bar_spacing
        sp_right = sp_center + 2 * bar_spacing
        ax.axvspan(sp_left, sp_right, facecolor=policy_bg_colors["sp"], edgecolor=None)

        # General Policy mean bars
        for mi, metric in enumerate(metric_order):
            x = gp_center + (mi - 1.5) * bar_spacing
            val = final_general[metric]
            if val == 0:
                ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(x, val, width=bar_width, color=colors[metric], alpha=0.9)

        # Specialized Policy mean bars
        for mi, metric in enumerate(metric_order):
            x = sp_center + (mi - 1.5) * bar_spacing
            val = final_special[metric]
            if val == 0:
                ax.text(x, 0.01, "X", ha="center", va="bottom", fontsize=9)
            else:
                ax.bar(x, val, width=bar_width, color=colors[metric], alpha=0.9)


        # ------------------------------------------------------------
        #   ASSE X, TITOLO, LEGENDA
        # ------------------------------------------------------------
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=12)
        ax.set_xlim(group_spacing - 2, group_spacing * (n_drones + 1) + 2)


        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_title(
            "General Policy vs Specialized Policy — Normalized Metrics per Drone",
            fontsize=15)

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=colors[m])
            for m in metric_order
        ]
        ax.legend(legend_patches, legend_labels,
                  title="Metrics (normalized)")
        
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        # ------------------------------------------------------------
        #   SALVA PNG
        # ------------------------------------------------------------
        fig.savefig(self.save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\nPlot salvato in: {os.path.abspath(self.save_path)}")


    def plot_bix3_vs_mean(self, save_path="drone_plot_bix3.png"):

        metric_order = ["speed", "negE", "prog", "reward"]

        colors = {
            "speed":  "#1f77b4",
            "negE":   "#ff7f0e",
            "prog":   "#2ca02c",
            "reward": "#d62728",
        }
        policy_bg_colors = {
            "gp": (0.2, 0.4, 1.0, 0.08),
            "sp": (1.0, 0.3, 0.3, 0.08),
        }

        # ------------------------------------------------------------
        #   CALCOLO IDENTICO AL PLOT PRINCIPALE
        # ------------------------------------------------------------
        general_bar_values = {m: [] for m in metric_order}
        special_bar_values = {m: [] for m in metric_order}
        per_drone_GP = []
        per_drone_SP = []

        for _, row in self.df.iterrows():
            metrics = self._process_row(row)
            gp = metrics["baseline"]
            sp = metrics["trained"]

            drone_gp = {}
            drone_sp = {}

            for m in metric_order:

                # GP
                gp_val = gp[m]
                if gp_val > 0:
                    general_bar_values[m].append(gp_val)
                    drone_gp[m] = gp_val
                else:
                    drone_gp[m] = 0

                # SP
                vals = sp[m]
                if m in ("speed", "negE"):
                    valid_mask = vals > 0
                else:
                    valid_mask = np.ones_like(vals, dtype=bool)

                valid_vals = vals[valid_mask]
                if len(valid_vals) > 0:
                    sp_mean = valid_vals.mean()
                    special_bar_values[m].append(sp_mean)
                    drone_sp[m] = sp_mean
                else:
                    drone_sp[m] = 0

            per_drone_GP.append(drone_gp)
            per_drone_SP.append(drone_sp)

        # MEDIE FINALI
        final_general = {m: np.mean(general_bar_values[m]) for m in metric_order}
        final_special = {m: np.mean(special_bar_values[m]) for m in metric_order}

        # Trova BIX3
        bix_index = 14
        bix_gp = per_drone_GP[bix_index]
        bix_sp = per_drone_SP[bix_index]

        # ------------------------------------------------------------
        #   PLOT
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

        # linea progress
        progress_y = 300/1000
        ax.axhline(progress_y, color="#2ca02c", linestyle="--", linewidth=1.3, alpha=0.8)

        group_spacing = 3.0
        gp_sp_gap = 1.1
        bar_spacing = 0.22
        bar_width = 0.12

        xticks = []
        xticklabels = []

        groups = [
            ("BIX3", bix_gp, bix_sp),
            ("MEAN", final_general, final_special)
        ]

        # Loop corretto
        for idx, (label, gp_vals, sp_vals) in enumerate(groups, start=1):

            base_x = idx * group_spacing
            gp_center = base_x - gp_sp_gap / 2
            sp_center = base_x + gp_sp_gap / 2
            # Label sotto GP/SP
            ax.text(gp_center, -0.08, "General", ha="center", va="top",
                    fontsize=10, transform=ax.get_xaxis_transform())

            ax.text(sp_center, -0.08, "Specialized", ha="center", va="top",
                    fontsize=10, transform=ax.get_xaxis_transform())

            xticks.append(base_x)
            xticklabels.append(label)

            # sfondi GP-SP
            ax.axvspan(gp_center - 2 * bar_spacing, gp_center + 2 * bar_spacing,
                    facecolor=policy_bg_colors["gp"], edgecolor=None)
            ax.axvspan(sp_center - 2 * bar_spacing, sp_center + 2 * bar_spacing,
                    facecolor=policy_bg_colors["sp"], edgecolor=None)

            # ----- GP BARS + PALLINI -----
            for mi, m in enumerate(metric_order):
                x = gp_center + (mi - 1.5) * bar_spacing
                val = gp_vals[m]

                if val > 0:
                    ax.bar(x, val, width=bar_width, color=colors[m])
                else:
                    ax.text(x, 0.01, "X", ha="center")

                # Pallini raw SOLO per BIX3
                if label == "BIX3":
                    raw_val = self._process_row(self.df.iloc[bix_index])["baseline"][m]
                    if raw_val > 0:
                        ax.scatter(x, raw_val, color=colors[m], edgecolor="black", alpha=0.8)

            # ----- SP BARS + PALLINI -----
            for mi, m in enumerate(metric_order):
                x = sp_center + (mi - 1.5) * bar_spacing
                val = sp_vals[m]

                if val > 0:
                    ax.bar(x, val, width=bar_width, color=colors[m])
                else:
                    ax.text(x, 0.01, "X", ha="center")

                # Pallini raw SOLO per BIX3
                if label == "BIX3":
                    raw_vals = self._process_row(self.df.iloc[bix_index])["trained"][m]

                    if m in ("speed", "negE"):
                        mask = raw_vals > 0
                    else:
                        mask = np.ones_like(raw_vals, bool)

                    xs = np.full(len(raw_vals), x)
                    ax.scatter(xs[mask], raw_vals[mask],
                            color=colors[m], edgecolor="black", alpha=0.8)

        # final touches
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("Normalized Value")
        ax.set_title("BIX3 vs Mean Performance — Normalized Metrics")
        legend_labels = [
            "Speed (/24)",
            "Cost of Transport (/1)",
            "Progress (/1000)",
            "Reward (/1000)",
        ]

        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, color=colors[m])
            for m in metric_order
        ]

        ax.legend(
            legend_patches,
            legend_labels,
            title="Metrics (normalized)",
            fontsize=10
        )
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("\nPlot BIX3 vs Mean salvato in:", os.path.abspath(save_path))



if __name__ == "__main__":
    csv_path = "/home/andrea/Documents/Genesis/src/data_processing/foundation_eval_lean.csv"  # Sostituisci con il percorso reale del CSV
    plotter = URDFHistogramPlotter(csv_path)
    plotter.plot()
    plotter.plot_bix3_vs_mean()