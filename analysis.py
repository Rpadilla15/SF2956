import pandas as pd
import numpy as np
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.lines import Line2D
import matplotlib
import warnings
import os
from data_cleaning import (
    calculate_indicator_summary, 
    summarize_data_wide, 
    input_missing_data, 
    standardize)
from homology import (
    safe_linspace,
    stable_rank,
    betti_curve,
    safe_interp
)

# ------------------------------
# 0. Stop warnings
# ------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ------------------------------
# 1. Load data
# ------------------------------
data_dir = "C:/Users/padil/Documents/TDA/DATA/"

data_countries = pd.read_csv(os.path.join(data_dir, "Data_countries_extended.csv")) # Data for countries
data_regions = pd.read_csv(os.path.join(data_dir, "Data_regions_extended.csv")) # Data for regions
world_regions_latest = pd.read_csv(os.path.join(data_dir,"clean_world_regions.csv")) # Map of countries into regions
happiness_raw = pd.read_csv(os.path.join(data_dir,"Happiness_data.csv"), encoding='latin-1') 
democracy_raw = pd.read_csv(os.path.join(data_dir,"democracy-index-eiu.csv"), ) 


democracy = democracy_raw.loc[democracy_raw['Code'].notna() & (democracy_raw['Year'] == 2024.0)].copy()
democracy['Democracy'] = pd.cut(x=democracy['Democracy index'], bins=[0, 4, 6, 8, 10],
                     labels=[0,1,2,3])
democracy['Democracy label'] = pd.cut(x=democracy['Democracy index'], bins=[0, 4, 6, 8, 10],
                     labels=[ "authoritarian regime", "hybrid regime", "flawed democracy","full democracy"])

# ------------------------------
# 2. Mappings
# ------------------------------
indicators = {
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
    'NY.GDP.MKTP.KD.ZG': 'gdp_growth',
    'SP.DYN.LE00.IN': 'life_expectancy',
    'SP.DYN.IMRT.IN': 'infant_mortality',
    'SE.SEC.ENRR': 'school_enrollment',
    #'SE.ADT.LITR.ZS': 'literacy_rate',
    'SH.XPD.CHEX.GD.ZS': 'health_spending',
    #'SI.POV.GINI': 'gini',
    'SI.POV.DDAY': 'poverty',
    'SP.POP.GROW': 'pop_growth',
    'EG.ELC.ACCS.ZS': 'electricity_access',
    'SM.POP.NETM': 'migration',
    'NV.AGR.TOTL.ZS': 'agriculture_weight_GDP',
    'SE.XPD.TOTL.GD.ZS': 'education spending'
}

wb_region_map = {
    "Latin America and Caribbean (WB)": "Latin America & Caribbean",
    "Sub-Saharan Africa (WB)": "Sub-Saharan Africa",
    "East Asia and Pacific (WB)": "East Asia & Pacific",
    "Europe and Central Asia (WB)": "Europe & Central Asia",
    "Middle East, North Africa, Afghanistan and Pakistan (WB)": "Middle East, North Africa, Afghanistan & Pakistan",
    "South Asia (WB)": "South Asia",
    "North America (WB)": "North America"
}

# ------------------------------
# 3. Build data matrix by filling empty indicators and dropping non-filled enough countries
# ------------------------------
calculations = ['median', 'net_change']

# Country level data
data_wide = summarize_data_wide(data_countries, indicators, calculations)

country_region_map = world_regions_latest.set_index('Code')['World regions according to WB'].map(wb_region_map).to_dict()
data_wide['region'] = data_wide['Country Code'].map(country_region_map)

# Region level data
data_region_wide = summarize_data_wide(data_regions, indicators, calculations)
data_region_wide = data_region_wide.rename(columns={'Country Name': 'region'})

data_wide_full = input_missing_data(data_wide, data_region_wide)

if 'gdp_per_capita' in data_wide_full.columns: # Log-transform GDP per capita safely
    data_wide_full['gdp_per_capita'] = data_wide_full['gdp_per_capita'].apply(lambda x: np.log10(x) if x>0 else np.nan)

analysis_type = 1
if analysis_type==0:
    X, countries, regions = standardize(data_wide_full)

else:
    democracy_scores = democracy[['Code', 'Democracy', 'Democracy label']]

    democracy_data = data_wide_full.merge(
        democracy_scores,
        left_on='Country Code',  # Column from data_wide_full
        right_on='Code', # Column from happiness_scores
        how='inner'
    )
    label_dem = democracy_data['Democracy'].values
    dem_names = democracy_data['Democracy label'].values
    democracy_data['region'] = democracy_data['Democracy label']
    democracy_data = democracy_data.drop(columns=['Code', 'Democracy', 'Democracy label'])
    data_wide_full = democracy_data.copy()
    X, countries, regions = standardize(democracy_data)
    regions = dem_names



# ------------------------------
# 4. Persistent homology
# ------------------------------
# Global

unique_regions = sorted(set(regions))
cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(unique_regions))
region_colors = {region: cmap(i) for i, region in enumerate(unique_regions)}

hemisphere_colors = {}
for region in unique_regions:
    if region in ["East Asia & Pacific", "Latin America & Caribbean", "Middle East, North Africa, Afghanistan & Pakistan", "South Asia", "Sub-Saharan Africa"]:
        hemisphere_colors[region] = "black"
    elif region in ["Europe & Central Asia", "North America"]:
        hemisphere_colors[region] = "red"
    elif region in ["authoritarian regime", "hybrid regime"]:
        hemisphere_colors[region] = "black"
    elif region in ["flawed democracy","full democracy"]:
        hemisphere_colors[region] = "red"
    else:
        # fallback (optional)
        hemisphere_colors[region] = "gray"

metrics = ["euclidean"] # , "hamming", "mahalanobis"
for metric in metrics:
    dist_matrix = squareform(pdist(X, metric=metric))

    rips = ripser(dist_matrix, maxdim=1, distance_matrix=True)
    plt.figure()
    plot_diagrams(rips['dgms'], show=False, title="Persistence Diagrams")
    #plt.savefig(f"Persistence_{metric}.pdf", bbox_inches='tight')
    plt.close()

    homology_results = {}

    for dim, label in enumerate(["H0", "H1"]):
        diagram = rips["dgms"][dim]
        t_vals = safe_linspace(diagram, num=100, default_max=np.max(dist_matrix))

        # Stable Rank and Betti
        sr = stable_rank(diagram, t_vals, include_infinite=True)
        betti = betti_curve(diagram, t_vals)

        # Interpolate to a common t-grid (for comparison)
        t_uniform = np.linspace(0, np.max(dist_matrix), 100)
        sr_interp = safe_interp(t_uniform, t_vals, sr)
        betti_interp = safe_interp(t_uniform, t_vals, betti)

        homology_results[label] = {
            "diagram": diagram,
            "t_vals": t_vals,
            "sr": sr,
            "betti": betti,
            "t_uniform": t_uniform,
            "sr_uniform": sr_interp,
            "betti_uniform": betti_interp,
        }

    sr_H0 = homology_results["H0"]["sr_uniform"]
    betti_H0 = homology_results["H0"]["betti_uniform"]
    sr_H1 = homology_results["H1"]["sr_uniform"]
    betti_H1 = homology_results["H1"]["betti_uniform"]
    t_vals = homology_results["H0"]["t_uniform"]

    # Sampling
    number_data_points = X.shape[0]

    def uniform_distribution_interval(low, high):
        def f(distances):
            prob = np.clip((distances - low) / (high - low), 0, 1)
            if prob.sum() == 0:
                return np.ones_like(prob) / len(prob)
            return prob / prob.sum()
        return f

    distributions = {
        # "0_50": uniform_distribution_interval(0, 50),
        # "25_75": uniform_distribution_interval(25, 75),
        # "50_100": uniform_distribution_interval(50, 100),
        # "75_125": uniform_distribution_interval(75, 125),
        # "100_150": uniform_distribution_interval(100, 150),
        # "125_175": uniform_distribution_interval(125, 175),
        # "150_200": uniform_distribution_interval(150, 200),

        "0_4": uniform_distribution_interval(0, 4),
        "2_6": uniform_distribution_interval(2, 6),
        "4_8": uniform_distribution_interval(4, 8),
        "6_10": uniform_distribution_interval(6, 10),
        "8_12": uniform_distribution_interval(8, 12),
        "10_14": uniform_distribution_interval(10, 14),
    }

    number_instances = 100    # number of random samples per base point
    sample_size = 10          # number of points in each sample
    t_len = 100               # number of t-values
    t_global = np.linspace(0, np.max(dist_matrix), t_len)  # common t-grid

    sr_H0_samp, sr_H1_samp = {k: [] for k in distributions}, {k: [] for k in distributions}
    betti_H0_samp, betti_H1_samp = {k: [] for k in distributions}, {k: [] for k in distributions}

    # --- Main sampling loop ---
    for k, dist_func in distributions.items():
        print(f"Processing distribution: {k}")
        for i in range(number_data_points):
            p = dist_func(dist_matrix[i])

            # Generate multiple random subsets
            sampled_indices = []
            for _ in range(number_instances):
                sampled_indices.append(np.random.choice(number_data_points, size=sample_size, p=p))
            sampled_indices = np.unique(np.concatenate(sampled_indices))

            # Skip trivial subsets
            if len(sampled_indices) < 2:
                sr_H0_samp[k].append(np.zeros(t_len))
                sr_H1_samp[k].append(np.zeros(t_len))
                betti_H0_samp[k].append(np.zeros(t_len))
                betti_H1_samp[k].append(np.zeros(t_len))
                continue

            X_subset = X[sampled_indices]
            rips = ripser(X_subset, maxdim=1)
            H0_dgm, H1_dgm = rips['dgms']

            # H0 computations
            t_local_H0 = safe_linspace(H0_dgm, t_len, default_max=np.max(dist_matrix))
            sr0 = stable_rank(H0_dgm, t_local_H0)
            betti0 = betti_curve(H0_dgm, t_local_H0)
            sr_H0_samp[k].append(safe_interp(t_global, t_local_H0, sr0))
            betti_H0_samp[k].append(safe_interp(t_global, t_local_H0, betti0))

            # H1 computations
            t_local_H1 = safe_linspace(H1_dgm, t_len, default_max=np.max(dist_matrix))
            sr1 = stable_rank(H1_dgm, t_local_H1)
            betti1 = betti_curve(H1_dgm, t_local_H1)
            sr_H1_samp[k].append(safe_interp(t_global, t_local_H1, sr1))
            betti_H1_samp[k].append(safe_interp(t_global, t_local_H1, betti1))

    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, sr_H0, label="Stable Rank H0")
    plt.plot(t_vals, sr_H1, label="Stable Rank H1")
    plt.plot(t_vals, betti_H0, '--', label="Betti H0")
    plt.plot(t_vals, betti_H1, '--', label="Betti H1")
    plt.xlabel("t (filtration value)")
    plt.ylabel("Count")
    plt.title("Stable Rank and Betti Curves for H0 and H1")
    plt.legend()
    plt.show()


    # --- Distributions ---
    distribution_names = list(distributions.keys())
    n_distributions = len(distribution_names)
    number_samples = len(regions)

    # --- Homology / curves dictionaries ---
    homology_types = {
        'H0': sr_H0_samp,
        'H1': sr_H1_samp,
        'Betti0': betti_H0_samp,
        'Betti1': betti_H1_samp
    }
    homology_labels = ['H0', 'H1', 'Betti0', 'Betti1']

    # --- Initial settings ---
    current_homology = 'H1'
    current_distribution = 0  # index into distribution_names

    # --- Handle t=0 for log scale ---
    epsilon = 1e-5
    t_plot = np.maximum(t_global, epsilon)

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.3, left=0.2)
    ax.set_xscale('log')  # log scale
    ax.set_xlabel("t")
    ax.set_ylabel(current_homology)
    ax.grid(alpha=0.3)

    # --- Initial plot lines ---
    lines = []
    for i in range(number_samples):
        region = regions[i]
        curve_dict = homology_types[current_homology]
        curve = curve_dict[distribution_names[current_distribution]][i]
        #line, = ax.plot(t_plot, curve, color=region_colors[region], alpha=0.5, lw=0.6)
        line, = ax.plot(t_plot, curve, color=hemisphere_colors[region], alpha=0.5, lw=0.6)
        lines.append(line)

    ax.set_title(f"{current_homology} Curves - Distribution: {distribution_names[current_distribution]}")

    # --- Slider for distributions ---
    ax_slider = plt.axes([0.25, 0.15, 0.6, 0.03])
    slider = Slider(ax_slider, 'Distribution', 0, n_distributions-1,
                    valinit=current_distribution, valstep=1, valfmt='%0.0f')

    # --- Radio buttons for homology type ---
    ax_radio_hom = plt.axes([0.02, 0.5, 0.15, 0.2])
    radio_hom = RadioButtons(ax_radio_hom, homology_labels)

    # --- Update functions ---
    def update_slider(val):
        idx = int(slider.val)
        dist_name = distribution_names[idx]
        for i in range(number_samples):
            region = regions[i]
            curve_dict = homology_types[current_homology]
            lines[i].set_ydata(curve_dict[dist_name][i])
            #lines[i].set_color(region_colors[region])
            lines[i].set_color(hemisphere_colors[region])
        ax.set_title(f"{current_homology} Curves - Distribution: {dist_name}")
        fig.canvas.draw_idle()

    max_t_H0 = np.max([np.max(curve) for k in distributions.keys() for curve in sr_H0_samp[k]])
    max_t_H1 = np.max([np.max(curve) for k in distributions.keys() for curve in sr_H1_samp[k]])

    def update_homology(label):
        global current_homology
        current_homology = label
        ax.set_ylabel(current_homology)
        
        # --- x-axis scale ---
        if current_homology in ['H1', 'Betti1']:
            ax.set_xscale('log')
            ax.set_xlim(np.min(t_plot), np.max(t_plot))  # use t_plot for log
            pass
        else:
            ax.set_xscale('linear')
            if current_homology in ['H0']:
                ax.set_xlim(0, 1.5)
            elif current_homology in ['Betti0']:
                ax.set_xlim(0, 2.5)
            else:
                ax.set_xlim(0, np.max(t_global))
        
        # --- y-axis scaling ---
        if current_homology in ['H0', 'Betti0']:
            ax.set_ylim(0, np.max([np.max(sr_H0_samp[k][current_distribution]) for k in distributions]))
        else:  # H1 or Betti1
            ax.set_ylim(0, np.max([np.max(sr_H1_samp[k][current_distribution]) for k in distributions]))
        
        update_slider(slider.val)

    slider.on_changed(update_slider)
    radio_hom.on_clicked(update_homology)

    # --- Legend for regions ---
    #legend_elements = [Line2D([0],[0], color=region_colors[r], lw=2, label=r) for r in unique_regions]
    legend_elements = [Line2D([0],[0], color=hemisphere_colors[r], lw=2, label=r) for r in unique_regions]
    ax.legend(handles=legend_elements, title="Regions", bbox_to_anchor=(1.05,1), loc='upper left')

    plt.show()


    # ---------------------------------------------------------------------
    # 5. Dendrogram
    # ---------------------------------------------------------------------
    methods = ["complete", "average", "single", "ward"]
    for method in methods:
        linked = linkage(dist_matrix, method=method)

        threshold = 0.5 * max(linked[:, 2])  # threshold for coloring

        plt.figure(figsize=(35, 15))  # Big horizontal figure
        dend = dendrogram(
            linked,
            labels=countries,
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=threshold
        )

        ax = plt.gca()
        for lbl in ax.get_xmajorticklabels():
            country_name = lbl.get_text()
            region = data_wide_full.loc[data_wide_full['Country Name'] == country_name, 'region'].values[0]
            lbl.set_color(region_colors.get(region, 'black'))

        handles = [ # Legend for label colors beneath dendrogram
            Line2D([0], [0], marker='o', color='w', label=r,
                markerfacecolor=region_colors[r], markersize=10)
            for r in unique_regions
        ]

        # Place legend further down below the dendrogram
        plt.legend(handles=handles, title="World Bank Regions",
                loc='upper center', bbox_to_anchor=(0.5, -0.35),
                ncol=len(unique_regions), fancybox=True)

        plt.title("Hierarchical Clustering Dendrogram (Regions Colored)", fontsize=18)
        plt.ylabel("Distance", fontsize=14)
        plt.subplots_adjust(bottom=0.45, left=0.05, right=0.95)  # More space for legend

        # Optional: save to PDF for a full-page view
        #plt.savefig(f"Dendrogram_{metric}_{method}.pdf", bbox_inches='tight')

        plt.show()