import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

# Constants
TRAVEL_START = '2024-03-10'
TRAVEL_END = '2024-08-31'
DB_PATH = 'summary.db'  # Using relative path for portability

# Define time periods
CURRENT_DATE = '2025-03-28'


@st.cache_resource
def get_connection():
    """Create a database connection."""
    try:
        return sqlite3.connect(DB_PATH, check_same_thread=False)
    except sqlite3.Error as e:
        st.error(f"Failed to connect to database: {e}")
        return None

@st.cache_data
def load_data_from_db(query):
    """Returns a DataFrame based on the query using the cached connection."""
    try:
        conn = get_connection()
        if conn is None:
            return None
        df = pd.read_sql_query(query, conn)
        if df.empty:
            st.warning("No data found in the table.")
        return df
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        st.error(f"Error loading data: {e}")
        return None

# Build the Streamlit App
st.title("The Impact of Extended Travel on Health Metrics")

# Add project description
st.markdown("""
### Project Overview
This dashboard analyzes health metrics collected through my Garmin watch to understand the impact of extended travel on various health indicators. During a significant journey through South America and Europe from March to August 2024, I wanted to quantify how traveling affected my daily health patterns. While many perceive extended travel as a prolonged vacation, the health data reveals a different story - one of significant physical demands and physiological adaptation to constantly changing environments.

### Data Collection Methodology
The health metrics analyzed in this dashboard were collected using the GarminDB package, an open-source tool that interfaces with Garmin Connect. The data collection process involved:

* **Automated Data Retrieval:** Daily monitoring data was automatically downloaded from Garmin Connect, including all-day heart rate, activity levels, stress measurements, and intensity minutes.
* **Comprehensive Metrics:** The system extracted and stored detailed data points including:
    - Sleep patterns and quality
    - Resting heart rate
    - Daily step counts
    - Stress levels (derived from Heart Rate Variability)
* **Data Processing:** Raw data was processed and stored in a SQLite database, allowing for efficient querying and analysis


### Analysis Periods
The analysis compares key health metrics across three periods:
- **Before Travel**: Baseline health patterns
- **During Travel**: Health metrics while exploring new places
- **After Travel**: Post-travel recovery and readjustment

Through this data, we can observe how factors like steps, heart rate, resting heart rate, and stress levels fluctuate between regular routines and travel periods. This provides valuable insights into how our bodies adapt to changes in environment, activity, and daily patterns during extended travel.
""")


# Load daily data with specific columns
sql_query = """
    SELECT 
        day,
        CAST(steps AS FLOAT) as steps,
        CAST(hr_avg AS FLOAT) as hr_avg,
        CAST(hr_min AS FLOAT) as hr_min,
        CAST(hr_max AS FLOAT) as hr_max,
        CAST(stress_avg AS FLOAT) as stress_avg,
        CAST(rhr_avg AS FLOAT) as rhr_avg
    FROM days_summary 
    WHERE day >= '2023-10-01'
    ORDER BY day
"""

data = load_data_from_db(sql_query)

if data is not None:
    # Convert date to datetime
    data['day'] = pd.to_datetime(data['day'])
    
    # Replace any infinite values with NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # Clean invalid stress values (-1, -2) and other invalid data
    data.loc[data['stress_avg'] <= 0, 'stress_avg'] = np.nan
    data.loc[data['hr_avg'] <= 0, 'hr_avg'] = np.nan
    data.loc[data['rhr_avg'] <= 0, 'rhr_avg'] = np.nan
    data.loc[data['steps'] < 0, 'steps'] = np.nan
    
    # Drop rows where all metric columns are NaN
    metric_columns = ['steps', 'hr_avg', 'stress_avg', 'rhr_avg']
    data = data.dropna(subset=metric_columns, how='all')
    
    # Create period labels
    data['period'] = 'After Travel'
    data.loc[data['day'] < TRAVEL_START, 'period'] = 'Before Travel'
    data.loc[(data['day'] >= TRAVEL_START) & (data['day'] <= TRAVEL_END), 'period'] = 'During Travel'
    
    # Calculate period averages
    period_stats = data.groupby('period').agg({
        'steps': 'mean',
        'hr_avg': 'mean',
        'stress_avg': 'mean',
        'rhr_avg': 'mean'
    }).round(2)
    
    # Remove the period comparison table at the top
    st.subheader("Period Comparisons")
    
    # Create time series plots with detailed statistics
    metrics = {
        'steps': 'Daily Steps',
        'hr_avg': 'Average Heart Rate',
        'rhr_avg': 'Resting Heart Rate',
        'stress_avg': 'Average Stress Level'
    }
    
    for metric, title in metrics.items():
        st.write(f"\n### {title}")
        
        # Calculate detailed statistics for each period
        period_detailed_stats = data.groupby('period')[metric].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).round(2)
        
        # Create the time series plot
        fig = px.scatter(data, x='day', y=metric, color='period',
                        title=f'{title} Over Time',
                        labels={metric: title, 'day': 'Date'},
                        color_discrete_map={
                            'Before Travel': '#636EFA',
                            'During Travel': '#EF553B',
                            'After Travel': '#00CC96'
                        })
        
        # Add vertical lines for travel period
        fig.add_vline(x=TRAVEL_START, line_dash="dash", line_color="gray")
        fig.add_vline(x=TRAVEL_END, line_dash="dash", line_color="gray")
        
        # Add annotations for significant hikes in the steps chart
        if metric == 'steps':
            significant_hikes = {
                '2024-03-23': {'name': 'Patagonia', 'yshift': 40},
                '2024-04-24': {'name': 'Machu Picchu', 'yshift': 20},
                '2024-07-31': {'name': 'Morskie Oko, Poland', 'yshift': 20}
            }
            
            for date, hike_info in significant_hikes.items():
                hike_data = data[data['day'] == date]
                if not hike_data.empty:
                    steps_value = hike_data['steps'].iloc[0]
                    fig.add_annotation(
                        x=date,
                        y=steps_value,
                        text=hike_info['name'],
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor='rgba(0, 0, 0, 0.6)',
                        arrowwidth=2,
                        arrowsize=1,
                        bgcolor='rgba(255, 255, 255, 0.9)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1,
                        font=dict(size=10),
                        yshift=hike_info['yshift']
                    )
        
        # Add food poisoning annotation to heart rate chart
        if metric == 'hr_avg':
            illness_date = '2024-04-11'
            illness_data = data[data['day'] == illness_date]
            if not illness_data.empty:
                hr_value = illness_data['hr_avg'].iloc[0]
                fig.add_annotation(
                    x=illness_date,
                    y=hr_value,
                    text='Food Poisoning in Bolivia',
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='rgba(0, 0, 0, 0.6)',
                    arrowwidth=2,
                    arrowsize=1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=10),
                    yshift=30
                )

        # Add food poisoning annotation to resting heart rate chart
        if metric == 'rhr_avg':
            illness_date = '2024-04-15'
            illness_data = data[data['day'] == illness_date]
            if not illness_data.empty:
                rhr_value = illness_data['rhr_avg'].iloc[0]
                fig.add_annotation(
                    x=illness_date,
                    y=rhr_value,
                    text='Food Poisoning in Bolivia',
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='rgba(0, 0, 0, 0.6)',
                    arrowwidth=2,
                    arrowsize=1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=10),
                    yshift=30
                )

        # Add stress annotations
        if metric == 'stress_avg':
            # Pre-travel job stress annotation
            pre_travel_date = '2024-02-03'
            pre_travel_data = data[data['day'] == pre_travel_date]
            if not pre_travel_data.empty:
                stress_value = pre_travel_data['stress_avg'].iloc[0]
                fig.add_annotation(
                    x=pre_travel_date,
                    y=stress_value,
                    text='Pre-Travel Job Transition Stress',
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='rgba(0, 0, 0, 0.6)',
                    arrowwidth=2,
                    arrowsize=1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=10),
                    yshift=30
                )

            # First food poisoning incident
            illness_date = '2024-04-11'
            illness_data = data[data['day'] == illness_date]
            if not illness_data.empty:
                stress_value = illness_data['stress_avg'].iloc[0]
                fig.add_annotation(
                    x=illness_date,
                    y=stress_value,
                    text='Food Poisoning in Bolivia',
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='rgba(0, 0, 0, 0.6)',
                    arrowwidth=2,
                    arrowsize=1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=10),
                    yshift=30
                )
            
            # Second food poisoning incident
            illness_date = '2024-08-10'
            illness_data = data[data['day'] == illness_date]
            if not illness_data.empty:
                stress_value = illness_data['stress_avg'].iloc[0]
                fig.add_annotation(
                    x=illness_date,
                    y=stress_value,
                    text='Food Poisoning & Long Travel Day',
                    showarrow=True,
                    arrowhead=1,
                    arrowcolor='rgba(0, 0, 0, 0.6)',
                    arrowwidth=2,
                    arrowsize=1,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=10),
                    yshift=30
                )
        
        # Update layout for better readability
        fig.update_layout(
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig)
        
        # Display detailed statistics
        st.write("**Detailed Statistics by Period:**")
        
        # Format the step numbers that need comma formatting
        stats_df = period_detailed_stats.copy()
        if metric == 'steps':
            stats_df['mean'] = stats_df['mean'].apply(lambda x: f"{x:,.0f}")
            stats_df['std'] = stats_df['std'].apply(lambda x: f"{x:,.0f}")
            stats_df['min'] = stats_df['min'].apply(lambda x: f"{x:,.0f}")
            stats_df['max'] = stats_df['max'].apply(lambda x: f"{x:,.0f}")
        else:
            stats_df = stats_df.round(2)
        
        # Rename columns for better readability
        stats_df.columns = ['Mean', 'Std Dev', 'Min', 'Max', 'Count']
        st.dataframe(stats_df)
        
        # Perform statistical test between travel and post-travel periods
        travel_stats = period_detailed_stats.loc['During Travel']
        post_travel_stats = period_detailed_stats.loc['After Travel']
        
        if not (pd.isna(travel_stats['mean']) or pd.isna(post_travel_stats['mean'])):
            # Get statistics
            n1 = travel_stats['count']
            n2 = post_travel_stats['count']
            mean1 = travel_stats['mean']
            mean2 = post_travel_stats['mean']
            std1 = travel_stats['std']
            std2 = post_travel_stats['std']
            
            # Calculate z-score
            pooled_se = np.sqrt((std1**2/n1) + (std2**2/n2))
            z_score = (mean1 - mean2) / pooled_se
            
            # Calculate two-tailed p-value using standard normal distribution
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Display statistical significance
            st.write("\n**Statistical Significance:**")
            st.write(f"Comparing 'During Travel' vs 'After Travel':")
            if metric == 'steps':
                st.write(f"- During Travel: mean = {mean1:,.0f}, std = {std1:,.0f}, n = {int(n1)}")
                st.write(f"- After Travel: mean = {mean2:,.0f}, std = {std2:,.0f}, n = {int(n2)}")
            else:
                st.write(f"- During Travel: mean = {mean1:.2f}, std = {std1:.2f}, n = {int(n1)}")
                st.write(f"- After Travel: mean = {mean2:.2f}, std = {std2:.2f}, n = {int(n2)}")
            st.write(f"- Pooled standard error: {pooled_se:.3f}")
            st.write(f"- z-score: {z_score:.2f}")
            st.write(f"- p-value: {p_value:.4f}")
            significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
            st.write(f"- The difference in means is {significance} (α = 0.05)")
        else:
            st.write("\n**Statistical Significance:**")
            st.write("Insufficient data for statistical testing in one or both periods.")
        
        st.write("---")
    
    # Calculate and display interesting insights
    st.subheader("Key Insights")
    
    st.markdown("""
    ### Physical Activity Patterns
    During the travel period, significant changes in physical activity were observed:
    """)
    
    # Steps comparison
    travel_steps_avg = period_stats.loc['During Travel', 'steps']
    home_steps_avg = period_stats.loc['After Travel', 'steps']
    steps_increase = ((travel_steps_avg - home_steps_avg) / home_steps_avg * 100)
    
    # Calculate number of high step days during travel
    travel_data = data[(data['day'] >= TRAVEL_START) & (data['day'] <= TRAVEL_END)]
    high_step_days = len(travel_data[travel_data['steps'] > 20000])
    
    st.write(f"- Daily steps averaged {travel_steps_avg:,.0f} during travel compared to {home_steps_avg:,.0f} after travel")
    st.write(f"- This represents a {abs(steps_increase):.1f}% {'increase' if steps_increase > 0 else 'decrease'} in daily steps during travel")
    st.write(f"- During travel, there were {high_step_days} days where daily steps exceeded 20,000, highlighting the intense physical activity during city exploration and hiking adventures")
    st.write("- The range of daily steps was much wider during travel, from rest days with minimal movement to intense hiking days exceeding 30,000 steps, showing the varied nature of travel activities.")
    
    st.markdown("""
    ### Cardiovascular Health Indicators
    The analysis of heart rate metrics reveals interesting patterns in cardiovascular response to travel, particularly in challenging environments and varying conditions:
    """)
    
    # Heart rate insights
    hr_travel_avg = period_stats.loc['During Travel', 'hr_avg']
    hr_home_avg = period_stats.loc['After Travel', 'hr_avg']
    hr_change = ((hr_travel_avg - hr_home_avg) / hr_home_avg * 100)
    
    # Resting heart rate insights
    rhr_travel_avg = period_stats.loc['During Travel', 'rhr_avg']
    rhr_home_avg = period_stats.loc['After Travel', 'rhr_avg']
    rhr_change = ((rhr_travel_avg - rhr_home_avg) / rhr_home_avg * 100)
    
    # Calculate number of high RHR days during travel
    high_rhr_days = len(travel_data[travel_data['rhr_avg'] > 60])
    
    st.write(f"- Average heart rate was {hr_travel_avg:.1f} bpm during travel vs {hr_home_avg:.1f} bpm after travel")
    st.write(f"- This shows a {abs(hr_change):.1f}% {'increase' if hr_change > 0 else 'decrease'} in average heart rate during travel")
    st.write(f"- Resting heart rate averaged {rhr_travel_avg:.1f} bpm during travel compared to {rhr_home_avg:.1f} bpm after")
    st.write(f"- This indicates a {abs(rhr_change):.1f}% {'increase' if rhr_change > 0 else 'decrease'} in resting heart rate during travel")
    st.write(f"- The travel period included {high_rhr_days} days where resting heart rate surpassed 60 bpm, suggesting sustained periods of cardiovascular exertion and physiological adaptation")
    st.write("- Peak heart rates during travel were notably higher, particularly during challenging hikes at altitude and intense urban exploration days")
    
    
    st.markdown("""
    Several unique travel conditions likely contributed to these cardiovascular changes:

    **Sleep Quality & Altitude**
    
    The combination of varying sleep environments and altitude changes created unique challenges:

    * **Sleep Environment Variability:** Frequent transitions between sleeping arrangements impacted recovery, from overnight buses to unfomfortable accomodations. Varying bed sizes and temperatures (from cold mountain nights to humid coastal conditions) posed additional challenges.
    
    * **Altitude Adaptation:** Regular exposure to significant elevation changes in regions like the Andes Mountains required continuous cardiovascular adaptation. Overnight transportation and early departures further disrupted natural sleep cycles, particularly during multi-day treks.

    **Physical Demands**
    
    The physical challenges manifested through:

    
    * **Travel Intensity:** The travel style combined regular relocations every 3-4 days, extended periods of time carrying up to 40lb load, long periods of standing during transportation, and varied intensity from rest days to intense hiking. Recovery was limited by minimal rest days and constant adaptation to new physical challenges.
    
    * **Environmental Factors:** Activities were often complicated by high altitude conditions, variable weather, and challenging terrain from city streets to mountain paths.

    These conditions created a unique physiological environment where the cardiovascular system showed measurable responses to ongoing physical demands and environmental adaptations.
    """)
    
    st.markdown("""
    ### Stress and Adaptation
    The stress level measurements provide insights into how the body adapted to travel. Garmin calculates stress scores using Heart Rate Variability (HRV), which measures the time between heartbeats. Lower HRV (shorter intervals) indicates higher stress, while higher HRV suggests a more relaxed state.
    """)
    
    # Stress insights
    stress_travel_avg = period_stats.loc['During Travel', 'stress_avg']
    stress_home_avg = period_stats.loc['After Travel', 'stress_avg']
    stress_change = ((stress_travel_avg - stress_home_avg) / stress_home_avg * 100)
    
    st.write(f"- Average stress level was {stress_travel_avg:.1f} during travel and {stress_home_avg:.1f} after travel.")
    st.write(f"- This represents a {abs(stress_change):.1f}% {'increase' if stress_change > 0 else 'decrease'} in stress during travel.")
    st.write("- Peak stress levels during travel coincided with specific events like food poisoning and challenging travel days.")
    st.write("- Despite higher average stress, there were also periods of remarkably low stress during travel, particularly during extended stays in single locations")

    st.markdown("""
    **Travel-Related Stress Factors**
    
    The HRV-based stress measurements help quantify how various aspects of travel affected physiological response:

    * **Navigation:** Constant adaptation to unfamiliar surroundings and overcoming language barriers in different countries.
    
    * **Circadian Disruption:** Regular adjustments to new time zones and changes in daily routines, affecting natural body rhythms.
    
    * **Sleep Variability:** Frequent changes in sleeping environments and schedules, from overnight buses to varying accommodation qualities.
    
    * **Logistical Demands:** Ongoing mental load from planning routes, booking accommodations, and managing day-to-day travel logistics.
    
    * **Physical Strain:** Regular physical exertion from carrying heavy backpacks through various terrains and climates.
    
    * **Environmental Adaptation:** Continuous physiological adjustments to changes in altitude, climate, and living conditions.

    * **Health Incidents:** Multiple instances of food poisoning significantly impacted stress levels.
        
    These factors combined to create a dynamic stress profile throughout the journey, with the body constantly working to maintain homeostasis in changing conditions.
    """)
    
    # Add conclusion section
    st.markdown("""
    ### Conclusion: The True Nature of Extended Travel

    While spending six months traversing South America and Europe was an incredibly rewarding and transformative experience, the data presented in this analysis reveals the physical and mental demands of long-term travel. The metrics paint a clear picture of the body's response to constant movement, adaptation, and stress:

    For those contemplating extended travel, this analysis serves as both an inspiration and a reality check. While the experiences are invaluable and life-changing, it's crucial to understand that long-term travel is not a vacation – it's an endeavor that requires physical preparation, mental resilience, and careful attention to your body's signals. The rewards of exploring diverse cultures, witnessing breathtaking landscapes, and pushing personal boundaries come with a measurable impact on your physiology that should be respected and prepared for.
    """)

    st.markdown("""
    ### Study Limitations

    **Statistical Considerations**

    While this analysis provides valuable insights into health changes during travel, there are important statistical limitations to consider:

    * **Variance Heterogeneity:** The use of pooled standard error in our statistical tests assumes similar variances between travel and post-travel periods. However, our data shows notably different ranges and variability between these periods, particularly in:
        - Daily steps (wider range during travel due to alternating intense hiking days and rest days)
        - Heart rate measurements (greater variability during travel due to altitude changes and varying activity intensities)
        - Stress levels (more extreme fluctuations during travel)

    * **Sample Independence:** The sequential nature of the data (continuous daily measurements) may violate the assumption of independent samples, as health metrics often show temporal correlations.

    * **External Factors:** The analysis cannot fully account for all variables affecting health metrics, such as:
        - Seasonal changes between travel and post-travel periods
        - Different environmental conditions
        - Varying measurement conditions (e.g., device placement, daily wear time)

    These limitations suggest that the statistical significance values should be interpreted cautiously, with emphasis on the practical significance of the observed changes rather than purely statistical measures.
    """)

else:
    st.error("Failed to load data from the database.")
