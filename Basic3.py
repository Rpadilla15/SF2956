import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ripser import ripser
from persim import plot_diagrams
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.cm as cm
import warnings

# ------------------------------
# 0. Stop warnings
# ------------------------------
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ------------------------------
# 1. Load data
# ------------------------------
data_countries = pd.read_csv("Data_countries.csv") # Data for countries
data_regions = pd.read_csv("Data_regions.csv") # Data for regions
world_regions = pd.read_csv("world_regions.csv") # Map of countries into regions

# ------------------------------
# 2. Define indicators
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
    'EG.ELC.ACCS.ZS': 'electricity_access'
}

# ------------------------------
# 3. Build data matrix by filling empty indicators and dropping non-filled enough countries
# ------------------------------
country_data = data_countries[data_countries['Series Code'].isin(indicators.keys())].copy()
year_cols = country_data.columns[4:]  # all year columns
country_data[year_cols] = country_data[year_cols].apply(pd.to_numeric, errors='coerce')
country_data['value'] = country_data[year_cols].median(axis=1)
country_data['indicator'] = country_data['Series Code'].map(indicators)

data_wide = country_data.pivot_table(
    index=['Country Name', 'Country Code'],
    columns='indicator',
    values='value',
    aggfunc='median'
).reset_index()

world_regions_latest = world_regions[world_regions['Year'] == 2023].copy()
world_regions_latest['Code'] = world_regions_latest['Code'].str.strip()
world_regions_latest['World regions according to WB'] = world_regions_latest['World regions according to WB'].str.strip()

wb_region_map = {
    "Latin America and Caribbean (WB)": "Latin America & Caribbean",
    "Sub-Saharan Africa (WB)": "Sub-Saharan Africa",
    "East Asia and Pacific (WB)": "East Asia & Pacific",
    "Europe and Central Asia (WB)": "Europe & Central Asia",
    "Middle East, North Africa, Afghanistan and Pakistan (WB)": "Middle East, North Africa, Afghanistan & Pakistan",
    "South Asia (WB)": "South Asia",
    "North America (WB)": "North America"
}

country_region_map = world_regions_latest.set_index('Code')['World regions according to WB'].map(wb_region_map).to_dict()
data_wide['region'] = data_wide['Country Code'].map(country_region_map)

region_data = data_regions[data_regions['Series Code'].isin(indicators.keys())].copy()
region_data[year_cols] = region_data[year_cols].apply(pd.to_numeric, errors='coerce')
region_data['value'] = region_data[year_cols].median(axis=1)
region_data['indicator'] = region_data['Series Code'].map(indicators)

data_region_wide = region_data.pivot_table(
    index=['Country Name', 'Country Code'],
    columns='indicator',
    values='value',
    aggfunc='median'
).reset_index()
data_region_wide = data_region_wide.rename(columns={'Country Name': 'region'})

filled_countries = [] # Fills missing country indicators from region-level values
for ind in indicators.values():
    if ind not in data_wide.columns:
        continue
    missing_idx = data_wide[ind].isna()
    for idx in data_wide[missing_idx].index:
        region = data_wide.loc[idx, 'region']
        if pd.notna(region):
            if region in data_region_wide['region'].values:
                if ind in data_region_wide.columns:
                    region_row = data_region_wide[data_region_wide['region'] == region]
                    region_value = region_row[ind].values[0]
                    if pd.notna(region_value):
                        data_wide.at[idx, ind] = region_value
                        filled_countries.append((data_wide.loc[idx, 'Country Name'], ind, region))


print("\nFilled missing indicators from regions:")
for c, ind, r in filled_countries:
    print(f"{c}: {ind} filled from {r}")


indicator_cols = [i for i in indicators.values() if i in data_wide.columns]
missing_mask = data_wide[indicator_cols].isna()
dropped_info = data_wide[missing_mask.any(axis=1)].copy()
dropped_info['missing_indicators'] = dropped_info[indicator_cols].apply(
    lambda row: ', '.join(row.index[row.isna()]), axis=1
)
dropped_info = dropped_info[['Country Name', 'Country Code', 'missing_indicators']]

print("\nCountries dropped due to remaining missing indicators and which indicators were missing:")
print(dropped_info)

data_wide = data_wide.dropna(subset=indicator_cols)

if 'gdp_per_capita' in data_wide.columns: # Log-transform GDP per capita safely
    data_wide['gdp_per_capita'] = data_wide['gdp_per_capita'].apply(lambda x: np.log10(x) if x>0 else np.nan)

numeric_cols = [c for c in data_wide.columns if c not in ['Country Name', 'Country Code', 'region']]
X = StandardScaler().fit_transform(data_wide[numeric_cols])
countries = data_wide['Country Name'].values
regions = data_wide['region'].values












# ------------------------------
# 4. Persistent homology
# ------------------------------
# Global

def safe_linspace(diagram, num=100, default_max=1.0):
    """Return a safe linspace for t values based on finite deaths."""
    if diagram.size == 0:
        return np.linspace(0, default_max, num)
    finite_mask = np.isfinite(diagram[:, 1])
    if not finite_mask.any():
        return np.linspace(0, default_max, num)
    max_val = np.max(diagram[finite_mask][:, 1])
    if max_val == 0 or not np.isfinite(max_val):
        max_val = default_max
    return np.linspace(0, max_val, num)


def stable_rank(diagram, t_values, include_infinite=True):
    """Compute stable rank curve (count of intervals with lifetime > 2t)."""
    if diagram.size == 0:
        return np.zeros_like(t_values)

    finite_births = np.isfinite(diagram[:, 0])
    if include_infinite:
        dgm = diagram[finite_births]
    else:
        finite_deaths = np.isfinite(diagram[:, 1])
        dgm = diagram[finite_births & finite_deaths]

    if dgm.size == 0:
        return np.zeros_like(t_values)

    births, deaths = dgm[:, 0], dgm[:, 1]
    lifetimes = deaths - births
    lifetimes = np.nan_to_num(lifetimes, posinf=np.inf)

    return np.array([np.sum(lifetimes > 2 * t) for t in t_values])


def betti_curve(diagram, t_values):
    """Compute Betti curve: number of features alive at each t."""
    if diagram.size == 0:
        return np.zeros_like(t_values, dtype=int)

    finite_births = np.isfinite(diagram[:, 0])
    dgm = diagram[finite_births]
    if dgm.size == 0:
        return np.zeros_like(t_values, dtype=int)

    births, deaths = dgm[:, 0], dgm[:, 1]
    return np.array([np.sum((births <= t) & (t < deaths)) for t in t_values], dtype=int)


def safe_interp(x_new, x_orig, y_orig):
    """Interpolate safely, handling constant y_orig arrays."""
    if np.all(y_orig == y_orig[0]):
        return np.full_like(x_new, y_orig[0])
    return np.interp(x_new, x_orig, y_orig)

unique_regions = sorted(set(regions))
cmap = cm.get_cmap("tab10", len(unique_regions))
region_colors = {region: cmap(i) for i, region in enumerate(unique_regions)}

hemisphere_colors = {}
for region in unique_regions:
    if region in ["East Asia & Pacific", "Latin America & Caribbean", "Middle East, North Africa, Afghanistan & Pakistan", "South Asia", "Sub-Saharan Africa"]:
        hemisphere_colors[region] = "black"
    elif region in ["Europe & Central Asia", "North America"]:
        hemisphere_colors[region] = "red"
    else:
        # fallback (optional)
        hemisphere_colors[region] = "gray"

metrics = ["cityblock", "euclidean", "cosine"] # , "hamming", "mahalanobis"
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
        "0_50": uniform_distribution_interval(0, 50),
        "25_75": uniform_distribution_interval(25, 75),
        "50_100": uniform_distribution_interval(50, 100),
        "75_125": uniform_distribution_interval(75, 125),
        "100_150": uniform_distribution_interval(100, 150),
        "125_175": uniform_distribution_interval(125, 175),
        "150_200": uniform_distribution_interval(150, 200),
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
            region = data_wide.loc[data_wide['Country Name'] == country_name, 'region'].values[0]
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

        #plt.show()