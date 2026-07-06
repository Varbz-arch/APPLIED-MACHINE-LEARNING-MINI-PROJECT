# suggestions.py

DISEASE_SUGGESTIONS = {
    'Typhoid': {
        'what_to_do': [
            'Take prescribed antibiotics (Azithromycin or Ciprofloxacin)',
            'Drink ORS (Oral Rehydration Solution) to prevent dehydration',
            'Get plenty of rest and sleep',
            'Eat easily digestible foods like khichdi, soup, bananas',
            'Maintain proper hygiene - wash hands frequently',
            'Avoid sharing food, drinks, or utensils with others'
        ],
        'medical_attention': [
            'High fever (above 103°F / 39.4°C) lasting more than 3-4 days',
            'Severe abdominal pain or bloating',
            'Blood in stool or vomit',
            'Difficulty breathing',
            'Signs of dehydration (dry mouth, no urination for 8+ hours)'
        ],
        'prevention': [
            'Get Typhoid vaccination before traveling to high-risk areas',
            'Drink only boiled or bottled water',
            'Avoid street food and raw vegetables',
            'Wash hands thoroughly with soap before eating',
            'Maintain proper sanitation and hygiene'
        ],
        'risk_factors': [
            'Living in or traveling to areas with poor sanitation',
            'Eating food prepared by someone who has typhoid',
            'Drinking contaminated water',
            'Close contact with an infected person'
        ]
    },
    
    'Cholera': {
        'what_to_do': [
            'Immediate rehydration with ORS (Oral Rehydration Solution)',
            'Seek medical help immediately - cholera can be fatal within hours',
            'Continue breastfeeding if infant',
            'Zinc supplements may help reduce severity',
            'Strict isolation to prevent spread'
        ],
        'medical_attention': [
            'Severe watery diarrhea (more than 3-4 episodes per hour)',
            'Signs of severe dehydration (sunken eyes, wrinkled skin)',
            'Very low blood pressure or fainting',
            'Inability to drink or keep fluids down',
            'Muscle cramps due to electrolyte loss'
        ],
        'prevention': [
            'Drink only safe/boiled water',
            'Wash hands frequently with soap',
            'Eat thoroughly cooked hot food',
            'Avoid raw fruits and vegetables unless peeled',
            'Get oral cholera vaccine if traveling to endemic areas'
        ],
        'risk_factors': [
            'Living in areas with poor sanitation',
            'Drinking untreated water',
            'Eating raw or undercooked seafood',
            'Low stomach acid (elderly, children, acid-reducing medications)'
        ]
    },
    
    'Malaria': {
        'what_to_do': [
            'Take prescribed antimalarial medications (Artemisinin-based)',
            'Use paracetamol to reduce fever',
            'Get plenty of rest',
            'Stay hydrated - drink water and ORS',
            'Complete the full course of medication'
        ],
        'medical_attention': [
            'High fever with chills and sweating',
            'Severe headache and body aches',
            'Confusion or seizures',
            'Difficulty breathing',
            'Yellowing of skin/eyes (jaundice)'
        ],
        'prevention': [
            'Use mosquito nets while sleeping',
            'Apply mosquito repellent (DEET, Picaridin)',
            'Wear long sleeves and pants during evening/night',
            'Eliminate standing water around home',
            'Take prophylactic medication before traveling to high-risk areas'
        ],
        'risk_factors': [
            'Living in or traveling to tropical/subtropical regions',
            'Not using mosquito protection',
            'Outdoor activities during dusk/dawn',
            'Lack of immunity (children, pregnant women, travelers)'
        ]
    },
    
    'Dengue': {
        'what_to_do': [
            'Take paracetamol for fever (AVOID Aspirin and Ibuprofen)',
            'Drink plenty of fluids - water, ORS, coconut water',
            'Get complete bed rest',
            'Monitor platelet count regularly',
            'Seek hospital care if warning signs appear'
        ],
        'medical_attention': [
            'Severe abdominal pain or tenderness',
            'Persistent vomiting (more than 3 times in 24 hours)',
            'Bleeding from nose or gums',
            'Blood in vomit or stool',
            'Extreme fatigue or restlessness',
            'Decrease in platelet count below 50,000'
        ],
        'prevention': [
            'Eliminate mosquito breeding sites (standing water)',
            'Use mosquito repellents and nets',
            'Wear full-sleeve clothing',
            'Use mosquito screens on windows',
            'Community clean-up drives to remove stagnant water'
        ],
        'risk_factors': [
            'Living in tropical and subtropical areas',
            'Previous dengue infection (higher risk of severe dengue)',
            'Poor mosquito control in the area',
            'Weakened immune system'
        ]
    },
    
    'Heatstroke': {
        'what_to_do': [
            'Move to a cool, shaded or air-conditioned area immediately',
            'Remove excess clothing',
            'Apply cool water to skin and fan vigorously',
            'Apply ice packs to neck, armpits, and groin',
            'Drink cool water if conscious and able to swallow',
            'THIS IS A MEDICAL EMERGENCY - Call ambulance immediately'
        ],
        'medical_attention': [
            'Body temperature above 104°F (40°C)',
            'Confusion, slurred speech, or seizures',
            'Loss of consciousness',
            'Hot, dry skin (no sweating despite heat)',
            'Rapid, strong pulse or rapid shallow breathing'
        ],
        'prevention': [
            'Stay hydrated - drink water every 15-20 minutes',
            'Avoid outdoor activities during peak heat (11 AM - 4 PM)',
            'Wear light-colored, loose, breathable clothing',
            'Take frequent breaks in shade or AC',
            'Never leave children or pets in parked cars'
        ],
        'risk_factors': [
            'Elderly adults and young children',
            'Outdoor workers and athletes',
            'People with chronic diseases (heart disease, obesity)',
            'Taking certain medications (diuretics, antihistamines)',
            'Sudden exposure to high heat (heat waves)'
        ]
    }
}

# General tips for all diseases
GENERAL_TIPS = {
    'when_to_see_doctor': [
        'Fever lasting more than 3 days',
        'Severe headache or body pain',
        'Difficulty breathing',
        'Blood in stool, vomit, or urine',
        'Signs of dehydration (dry mouth, no urination for 8+ hours)',
        'Confusion or disorientation'
    ],
    'emergency_helplines': {
        'Ambulance': '108',
        'Health Helpline': '1800-180-1104',
        'COVID Helpline': '1075'
    }
}
# Add this at the end of suggestions.py
if __name__ == '__main__':
    print("="*50)
    print("TESTING SUGGESTIONS.PY")
    print("="*50)
    
    for disease, info in DISEASE_SUGGESTIONS.items():
        print(f"\n📋 {disease}")
        print(f"   What to do: {info['what_to_do'][0][:50]}...")
        print(f"   Medical attention: {info['medical_attention'][0][:50]}...")
    
    print("\n✅ Suggestions.py loaded successfully!")
    print(f"📊 Total diseases covered: {len(DISEASE_SUGGESTIONS)}")