import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import folium
from streamlit_folium import folium_static
from typing import Dict, List, Optional

# Page config
st.set_page_config(
    page_title="Parkinson's Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }
    div.block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .custom-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #1f2937;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Location Manager Class
class LocationManager:
    def __init__(self):
        self.centers_data = {
            'Parkinsons Treatment Centers': [
                {
                    'name': 'NIMHANS - National Institute of Mental Health and Neurosciences',
                    'address': 'Hosur Road, Near Dairy Circle',
                    'area': 'Bangalore South',
                    'city': 'Bengaluru',
                    'state': 'Karnataka',
                    'lat': 12.9374,
                    'lon': 77.5958,
                    'specialties': ['Movement Disorders', 'Neurology', 'DBS Surgery', 'Research'],
                    'phone': '080-26995000',
                    'description': 'Premier neurological institute with specialized Parkinson\'s treatment unit'
                },
                {
                    'name': 'Manipal Hospital',
                    'address': '98, HAL Old Airport Road',
                    'area': 'Kodihalli',
                    'city': 'Bengaluru',
                    'state': 'Karnataka',
                    'lat': 12.9583,
                    'lon': 77.6408,
                    'specialties': ['Movement Disorders', 'Neurology', 'Rehabilitation'],
                    'phone': '080-25023355',
                    'description': 'Comprehensive neurology center with advanced Parkinson\'s treatment facilities'
                },
                {
                    'name': 'Apollo Hospital',
                    'address': '154/11, Opp. IIM Bangalore',
                    'area': 'Bannerghatta Road',
                    'city': 'Bengaluru',
                    'state': 'Karnataka',
                    'lat': 12.8918,
                    'lon': 77.6014,
                    'specialties': ['Neurology', 'Movement Disorders', 'Physical Therapy'],
                    'phone': '080-43561234',
                    'description': 'Specialized movement disorders clinic with multidisciplinary approach'
                },
                {
                    'name': 'Columbia Asia Hospital',
                    'address': '26/1, Dr. Rajkumar Road',
                    'area': 'Malleswaram',
                    'city': 'Bengaluru',
                    'state': 'Karnataka',
                    'lat': 13.0159,
                    'lon': 77.5555,
                    'specialties': ['Neurology', 'Physical Therapy', 'Rehabilitation'],
                    'phone': '080-39898969',
                    'description': 'Dedicated neurology department with focus on movement disorders'
                },
                {
                    'name': 'Fortis Hospital',
                    'address': '154/9, Bannerghatta Road',
                    'area': 'Bangalore South',
                    'city': 'Bengaluru',
                    'state': 'Karnataka',
                    'lat': 12.8898,
                    'lon': 77.5990,
                    'specialties': ['Movement Disorders', 'DBS Surgery', 'Rehabilitation'],
                    'phone': '080-66214444',
                    'description': 'Advanced neurological care center with DBS surgery facilities'
                }
            ]
        }
        self.areas = self._extract_areas()
        
    def _extract_areas(self) -> List[str]:
        return list(set(center['area'] for center in self.centers_data['Parkinsons Treatment Centers']))
    
    def get_center_by_name(self, name: str) -> Optional[Dict]:
        return next(
            (center for center in self.centers_data['Parkinsons Treatment Centers'] 
             if center['name'] == name),
            None
        )
    
    def get_centers_in_area(self, area: str) -> List[Dict]:
        return [
            center for center in self.centers_data['Parkinsons Treatment Centers'] 
            if center['area'] == area
        ]
    
    def get_all_centers(self) -> List[Dict]:
        return self.centers_data['Parkinsons Treatment Centers']

# Map Creation Function
def create_center_map(selected_center: Dict, all_centers: List[Dict], radius_km: float = 5) -> folium.Map:
    m = folium.Map(
        location=[selected_center['lat'], selected_center['lon']],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add marker for selected center
    folium.Marker(
        [selected_center['lat'], selected_center['lon']],
        popup=folium.Popup(
            f"""
            <div style='width: 200px'>
                <b>{selected_center['name']}</b><br>
                {selected_center['address']}<br>
                <b>Specialties:</b><br>
                {', '.join(selected_center['specialties'])}<br>
                📞 {selected_center['phone']}
            </div>
            """,
            max_width=300
        ),
        icon=folium.Icon(color='red', icon='info-sign'),
        tooltip=selected_center['name']
    ).add_to(m)
    
    # Add markers for other centers
    for center in all_centers:
        if center['name'] != selected_center['name']:
            folium.Marker(
                [center['lat'], center['lon']],
                popup=folium.Popup(
                    f"""
                    <div style='width: 200px'>
                        <b>{center['name']}</b><br>
                        {center['address']}<br>
                        <b>Specialties:</b><br>
                        {', '.join(center['specialties'])}<br>
                        📞 {center['phone']}
                    </div>
                    """,
                    max_width=300
                ),
                icon=folium.Icon(color='blue', icon='info-sign'),
                tooltip=center['name']
            ).add_to(m)
    
    # Add coverage radius
    folium.Circle(
        [selected_center['lat'], selected_center['lon']],
        radius=radius_km * 1000,  # Convert km to meters
        color='red',
        fill=True,
        fillOpacity=0.1
    ).add_to(m)
    
    return m



# Custom header with emoji
st.title("Parkinson's Disease Prediction Tool")

# Add tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "ℹ️ About Parkinson's Disease", 
    "🔍 Prediction Tool", 
    "📊 Feature Information",
    "🏥 Find Treatment Centers"
])

with tab1:
    st.markdown("""
    <div class="info-card">
        <h2 style='color: #1f2937;'>Understanding Parkinson's Disease</h2>
        <p style='font-size: 1.1em; color: #4b5563;'>
        Parkinson's disease is a progressive nervous system disorder that affects movement. 
        The disease primarily affects dopamine-producing neurons in a specific area of the brain called substantia nigra.
        Symptoms develop gradually over years, and the progression of symptoms is often different from one person to another.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3 style='color: #1f2937;'>Common Symptoms 🔍</h3>
            <ul style='color: #4b5563;'>
                <li><strong>Motor Symptoms:</strong>
                    <ul>
                        <li>Tremor, mainly at rest</li>
                        <li>Bradykinesia (slowness of movement)</li>
                        <li>Limb rigidity</li>
                        <li>Gait and balance problems</li>
                    </ul>
                </li>
                <li><strong>Non-motor Symptoms:</strong>
                    <ul>
                        <li>Cognitive impairment</li>
                        <li>Depression and anxiety</li>
                        <li>Sleep disorders</li>
                        <li>Loss of sense of smell</li>
                        <li>Speech and swallowing problems</li>
                    </ul>
                </li>
                <li><strong>Early Symptoms:</strong>
                    <ul>
                        <li>Smaller handwriting</li>
                        <li>Loss of smell</li>
                        <li>Trouble sleeping</li>
                        <li>Trouble moving or walking</li>
                        <li>Constipation</li>
                        <li>A soft or low voice</li>
                        <li>Masked face (serious, depressed look)</li>
                        <li>Dizziness or fainting</li>
                        <li>Stooping or hunching over</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3 style='color: #1f2937;'>Risk Factors and Causes ⚠️</h3>
            <ul style='color: #4b5563;'>
                <li><strong>Age:</strong> Risk increases with age, typically around 60 years</li>
                <li><strong>Genetics:</strong>
                    <ul>
                        <li>Having a close relative with Parkinson's disease</li>
                        <li>Certain genetic mutations</li>
                    </ul>
                </li>
                <li><strong>Environmental Factors:</strong>
                    <ul>
                        <li>Exposure to toxins</li>
                        <li>Head trauma</li>
                        <li>Certain medications</li>
                        <li>Rural living and farming</li>
                    </ul>
                </li>
                <li><strong>Sex:</strong> Men are more likely to develop Parkinson's</li>
                <li><strong>Other Risk Factors:</strong>
                    <ul>
                        <li>Beta blocker and calcium channel blocker use</li>
                        <li>Agricultural work</li>
                        <li>Industrial work</li>
                        <li>Well water drinking</li>
                    </ul>
                </li>
            </ul>
        </div>
        
        <div class="info-card">
            <h3 style='color: #1f2937;'>Prevention and Management 🌟</h3>
            <ul style='color: #4b5563;'>
                <li><strong>Exercise regularly</strong></li>
                <li><strong>Healthy diet</strong> rich in antioxidants</li>
                <li><strong>Regular medical check-ups</strong></li>
                <li><strong>Stress management</strong></li>
                <li><strong>Social support and engagement</strong></li>
                <li><strong>Physical therapy</strong></li>
                <li><strong>Occupational therapy</strong></li>
                <li><strong>Speech therapy</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h3 style='color: #1f2937;'>Early Detection Through Voice Analysis 🎤</h3>
        <p style='color: #4b5563;'>
        Voice changes are increasingly recognized as one of the earliest indicators of Parkinson's disease. Studies have shown that up to 90% of people with Parkinson's experience speech and voice disorders, including:
        </p>
        <ul style='color: #4b5563;'>
            <li>Reduced vocal volume (hypophonia)</li>
            <li>Monotone speech (dysprosodia)</li>
            <li>Hoarse or breathy voice quality</li>
            <li>Imprecise articulation</li>
            <li>Speaking rate changes</li>
        </ul>
        <p style='color: #4b5563;'>
        This tool uses advanced acoustic analysis to detect subtle changes in voice patterns that may indicate early stages of Parkinson's disease.
        The analysis is based on multiple biomedical voice measurements and utilizes machine learning algorithms for prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="info-card">
        <h2 style='color: #1f2937;'>Voice Analysis Prediction Tool</h2>
        <p style='color: #4b5563;'>
        Upload a CSV file containing voice measurements to predict the likelihood of Parkinson's Disease.
        The file should contain all required voice measurement features listed in the Feature Information tab.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define the expected features in correct order
    EXPECTED_FEATURES = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 
        'spread1', 'spread2', 'D2', 'PPE'
    ]

    # Load the model
    @st.cache_resource
    def load_model():
        return pickle.load(open('parkinson_classifier_model.pkl', 'rb'))

    try:
        model = load_model()
        scaler = MinMaxScaler()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show raw data
            st.markdown("""
            <div class="info-card">
                <h3 style='color: #1f2937;'>Data Preview</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head())
            
            # Verify all required features are present
            missing_cols = set(EXPECTED_FEATURES) - set(df.columns)
            if missing_cols:
                st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
                st.stop()
            
            # Select only the required features in correct order
            df = df[EXPECTED_FEATURES]
            
            # Scale the features
            scaled_features = scaler.fit_transform(df)
            
            # Make predictions
            if st.button("Make Predictions"):
                with st.spinner("Analyzing voice measurements..."):
                    predictions = model.predict(scaled_features)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Row': range(1, len(predictions) + 1),
                        'Prediction': ["Has Parkinson's Disease" if pred == 1 
                                     else "Does not have Parkinson's Disease" 
                                     for pred in predictions]
                    })
                    
                    # Display results in a card
                    st.markdown("""
                    <div class="info-card">
                        <h3 style='color: #1f2937;'>Prediction Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(results_df)
                    
                    # Display summary statistics
                    st.markdown("""
                    <div class="info-card">
                        <h3 style='color: #1f2937;'>Analysis Summary</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    total_cases = len(predictions)
                    positive_cases = np.sum(predictions == 1)
                    negative_cases = np.sum(predictions == 0)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("""
                        <div class="metric-card">
                            <h4 style='color: #1f2937;'>Total Cases</h4>
                            <p style='font-size: 24px; color: #3b82f6;'>{}</p>
                        </div>
                        """.format(total_cases), unsafe_allow_html=True)
                    with col2:
                        st.markdown("""
                        <div class="metric-card">
                            <h4 style='color: #1f2937;'>Positive Cases</h4>
                            <p style='font-size: 24px; color: #ef4444;'>{}</p>
                        </div>
                        """.format(positive_cases), unsafe_allow_html=True)
                    with col3:
                        st.markdown("""
                        <div class="metric-card">
                            <h4 style='color: #1f2937;'>Negative Cases</h4>
                            <p style='font-size: 24px; color: #10b981;'>{}</p>
                        </div>
                        """.format(negative_cases), unsafe_allow_html=True)
                    
                    # Add visualization
                    st.markdown("""
                    <div class="info-card">
                        <h3 style='color: #1f2937;'>Distribution of Predictions</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.bar_chart(results_df['Prediction'].value_counts())
                    
                    st.markdown("""
                    <div class="info-card" style='background-color: #f0f9ff; border-left: 4px solid #3b82f6;'>
                        <h4 style='color: #1f2937;'>⚠️ Important Note</h4>
                        <p style='color: #4b5563;'>
                        This tool is for screening purposes only and should not be used as a definitive diagnosis. 
                        The results should be interpreted by healthcare professionals in conjunction with other clinical findings.
                        Please consult with a qualified healthcare professional for proper medical evaluation and diagnosis.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.markdown("""
    <div class="info-card">
        <h2 style='color: #1f2937;'>Voice Measurement Features</h2>
        <p style='color: #4b5563;'>
        The prediction model analyzes various acoustic features extracted from voice recordings. These measurements provide detailed information about different aspects of voice characteristics that may be affected by Parkinson's disease.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    feature_descriptions = {
        'MDVP Measurements 📊': [
            ("MDVP:Fo(Hz)", "Average vocal fundamental frequency", "The average rate of vibration of the vocal folds during speech"),
            ("MDVP:Fhi(Hz)", "Maximum fundamental frequency", "The highest frequency of vocal fold vibration recorded"),
            ("MDVP:Flo(Hz)", "Minimum fundamental frequency", "The lowest frequency of vocal fold vibration recorded"),
        ],
        'Jitter Measurements 📈': [
            ("MDVP:Jitter(%)", "Frequency variation percentage", "Measure of variation in fundamental frequency cycle-to-cycle"),
            ("MDVP:Jitter(Abs)", "Absolute jitter in microseconds", "Cycle-to-cycle variation in fundamental frequency in absolute terms"),
            ("MDVP:RAP", "Relative amplitude perturbation", "Measure of variability in pitch over three cycles"),
            ("MDVP:PPQ", "Five-point period perturbation quotient", "Measure of variability in pitch over five cycles"),
            ("Jitter:DDP", "Average absolute difference between cycles", "Average difference between consecutive differences in fundamental frequency")
        ],
        'Shimmer Measurements 📉': [
            ("MDVP:Shimmer", "Local shimmer", "Measure of variability in amplitude cycle-to-cycle"),
            ("MDVP:Shimmer(dB)", "Shimmer in decibels", "Log of the amplitude variation"),
            ("Shimmer:APQ3", "Three-point amplitude perturbation quotient", "Measure of variability in amplitude over three cycles"),
            ("Shimmer:APQ5", "Five-point amplitude perturbation quotient", "Measure of variability in amplitude over five cycles"),
            ("MDVP:APQ", "Amplitude perturbation quotient", "Measure of variability in amplitude over eleven cycles"),
            ("Shimmer:DDA", "Average absolute differences", "Average of differences between consecutive variations in amplitude")
        ],
        'Additional Measures 🔬': [
            ("NHR", "Noise-to-harmonics ratio", "Ratio of noise to harmonics in the voice signal"),
            ("HNR", "Harmonics-to-noise ratio", "Ratio of harmonics to noise in the voice signal"),
            ("RPDE", "Recurrence period density entropy", "Measure of uncertainty in pitch period estimation"),
            ("DFA", "Detrended fluctuation analysis", "Signal scaling exponent related to turbulent noise"),
            ("spread1", "Nonlinear measure 1", "Nonlinear measure of fundamental frequency variation"),
            ("spread2", "Nonlinear measure 2", "Nonlinear measure of fundamental frequency variation"),
            ("D2", "Correlation dimension", "Measure of complexity in the voice signal"),
            ("PPE", "Pitch period entropy", "Measure of impairment in controlling stable pitch")
        ]
    }
    
    for category, features in feature_descriptions.items():
        st.markdown(f"""
        <div class="info-card">
            <h3 style='color: #1f2937;'>{category}</h3>
            <table style='width: 100%; color: #4b5563;'>
                <tr>
                    <th style='padding: 8px; text-align: left; color: #1f2937;'>Feature</th>
                    <th style='padding: 8px; text-align: left; color: #1f2937;'>Name</th>
                    <th style='padding: 8px; text-align: left; color: #1f2937;'>Description</th>
                </tr>
                {''.join([f"<tr><td style='padding: 8px; font-weight: bold;'>{feature}</td><td style='padding: 8px;'>{name}</td><td style='padding: 8px;'>{description}</td></tr>" for feature, name, description in features])}
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card" style='background-color: #f0f9ff; border-left: 4px solid #3b82f6;'>
        <h4 style='color: #1f2937;'>🔍 Technical Details</h4>
        <p style='color: #4b5563;'>
        These acoustic measurements are obtained through specialized voice recording analysis software. Each measurement captures different aspects of voice characteristics that may be affected by Parkinson's disease. The combination of these features, when analyzed together using machine learning algorithms, helps in identifying subtle changes in voice that may indicate the presence of Parkinson's disease.
        </p>
        <p style='color: #4b5563;'>
        Voice analysis has emerged as a promising tool for early detection because:
        </p>
        <ul style='color: #4b5563;'>
            <li>Voice changes often occur in early stages of the disease</li>
            <li>Voice recording is non-invasive and can be done remotely</li>
            <li>Digital analysis provides objective measurements</li>
            <li>Multiple aspects of voice can be analyzed simultaneously</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="info-card">
        <h2 style='color: #1f2937;'>Find Parkinson's Disease Treatment Centers</h2>
        <p style='color: #4b5563;'>
        Locate specialized medical centers and hospitals that offer treatment for Parkinson's disease.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize location manager
    location_mgr = LocationManager()
    
    # Create columns for the hospital finder section
    map_col, info_col = st.columns([2, 1])
    
    with info_col:
        # Area and center selection
        selected_area = st.selectbox(
            "Select Area",
            location_mgr.areas
        )
        
        centers_in_area = location_mgr.get_centers_in_area(selected_area)
        selected_center_name = st.selectbox(
            "Select Treatment Center",
            [center['name'] for center in centers_in_area]
        )
        
        # Get selected center details
        selected_center = location_mgr.get_center_by_name(selected_center_name)
        
        if selected_center:
            st.markdown(f"""
            <div class="info-card">
                <h4 style='color: #1f2937;'>{selected_center['name']}</h4>
                <p style='color: #4b5563;'>
                <strong>Address:</strong><br>
                {selected_center['address']}<br>
                {selected_center['area']}<br>
                {selected_center['city']}, {selected_center['state']}
                </p>
                <p style='color: #4b5563;'>
                <strong>Phone:</strong><br>
                {selected_center['phone']}
                </p>
                <p style='color: #4b5563;'>
                <strong>Specialties:</strong><br>
                {', '.join(selected_center['specialties'])}
                </p>
                <p style='color: #4b5563;'>
                <strong>About:</strong><br>
                {selected_center['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional center resources section
            st.markdown("""
            <div class="info-card">
                <h4 style='color: #1f2937;'>Center Resources</h4>
                <ul style='color: #4b5563;'>
                    <li>24/7 Emergency Services</li>
                    <li>Movement Disorder Specialists</li>
                    <li>Support Groups</li>
                    <li>Rehabilitation Services</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with map_col:
        st.markdown("""
        <div class="info-card">
            <h3 style='color: #1f2937;'>Treatment Center Locations</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display map
        if selected_center:
            m = create_center_map(
                selected_center,
                location_mgr.get_all_centers()
            )
            folium_static(m)
            
            # Show map legend
            st.markdown("""
            <div style='background-color: #f3f4f6; padding: 10px; border-radius: 5px; margin-top: 10px;'>
                <p style='color: #4b5563; font-size: 0.9em; margin: 0;'>
                    🎯 <b>Map Legend:</b><br>
                    • Red Marker: Selected Center<br>
                    • Blue Markers: Other Centers<br>
                    • Red Circle: 5km coverage radius<br>
                    💡 Click markers for detailed information
                </p>
            </div>
            """, unsafe_allow_html=True)

    # Add treatment center resources
    st.markdown("""
    <div class="info-card">
        <h3 style='color: #1f2937;'>Treatment Center Resources</h3>
        <div style='color: #4b5563;'>
            <h4>What to Look for in a Treatment Center:</h4>
            <ul>
                <li><strong>Movement Disorder Specialists:</strong> Neurologists with specialized training in Parkinson's disease</li>
                <li><strong>Comprehensive Care Team:</strong> Including physical therapists, occupational therapists, and speech therapists</li>
                <li><strong>Support Services:</strong> Patient education, support groups, and counseling services</li>
                <li><strong>Research Opportunities:</strong> Access to clinical trials and new treatments</li>
                <li><strong>Accessibility:</strong> Location, transportation options, and scheduling flexibility</li>
            </ul>
        </div>
    </div>
    
    <div class="info-card">
        <h3 style='color: #1f2937;'>Preparing for Your Visit</h3>
        <div style='color: #4b5563;'>
            <h4>Checklist:</h4>
            <ul>
                <li>Medical records and test results</li>
                <li>Current medication list</li>
                <li>List of symptoms and changes since last visit</li>
                <li>Questions for your healthcare provider</li>
                <li>Insurance information</li>
                <li>Support person (family member or friend)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
# Add a footer
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 20px; margin-top: 30px;'>
    <p>Made with ❤️ for healthcare professionals and researchers</p>
    <p style='font-size: 0.8em;'>This tool is for screening purposes only. Please consult healthcare professionals for diagnosis.</p>
    <p style='font-size: 0.8em;'>© 2025 Parkinson's Disease Voice Analysis Tool</p>
</div>
""", unsafe_allow_html=True)