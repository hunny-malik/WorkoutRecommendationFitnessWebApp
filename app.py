import json, os
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

app = Flask(__name__, static_url_path='/static')

data_dir = os.path.join(os.path.dirname(__file__), 'data')

json_file_path = os.path.join(data_dir, 'exercises.json')
with open(json_file_path, 'r', encoding='utf-8') as file:
    exercises = json.load(file)

for exercise in exercises:
    images = exercise["images"]
    exercise["images"] = [image.split('/')[-1] for image in images]

dataframe = pd.DataFrame(exercises)

csv_file_path = os.path.join(data_dir, 'exercises.csv')
dataframe.to_csv(csv_file_path, index=False, sep=',')
csv_cleaned_file_path = os.path.join(data_dir, 'exercises_cleaned.csv')

df = pd.read_csv(csv_cleaned_file_path)

df['images'] = df['images'].apply(lambda x: [image.strip(" '") for image in x.strip("[]").split(", ")])

client = MongoClient("mongodb://localhost:27017/")
db = client["exercisesdb"]
collection = db["exercises"]

df_dict = df.to_dict(orient='records')
collection.insert_many(df_dict)

priority_fields = ['primaryMuscles','level', 'equipment', 'secondaryMuscles', 'force', 'mechanic', 'category']

priority_weights = [20, 15, 10, 5, 3, 2, 1]

df['content'] = df[priority_fields].apply(
    lambda row: (
        ' '.join([str(val) * weight for val, weight in zip(row, priority_weights)])
    ),
    axis=1
)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/welcome')
def welcome_page():
    return render_template('welcome.html')

@app.route('/beginner', methods=['GET', 'POST'])
def beginner():
    primary_muscles = ["Chest", "Biceps", "Abdominals", "Quadriceps", "Middle Back", "Glutes", "Hamstrings", "Calves "]
    selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle')
    if request.method == 'POST':
        selected_primary_muscle = request.form.get('selectedPrimaryMuscle')
        response = redirect(url_for('recommend_exercises'))
        response.set_cookie('selectedPrimaryMuscle', selected_primary_muscle)
        return response
    return render_template('beginner.html', primary_muscles=primary_muscles, selectedPrimaryMuscle=selected_primary_muscle)

@app.route('/advanced', methods=['GET', 'POST'])
def advanced():
    primary_muscles = ["Neck", "Shoulders", "Chest", "Biceps", "Forearms", "Abdominals", "Quadriceps", "Adductors", "Calves",
                       "Traps", "Triceps", "Lats", "Middle Back", "Lower Back", "Abductors", "Glutes", "Hamstrings", "Calves "]

    selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle')
    if request.method == 'POST':
        selected_primary_muscle = request.form.get('selectedPrimaryMuscle')
        response = redirect(url_for('recommend_exercises'))
        response.set_cookie('selectedPrimaryMuscle', selected_primary_muscle)
        return response
    return render_template('advanced.html', primary_muscles=primary_muscles, selectedPrimaryMuscle=selected_primary_muscle)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_exercises():
    exercise_data = []
    user_input = {}
    selected_primary_muscle= ""
    if request.method == 'POST':
        user_input = {field: request.form.get(field) for field in priority_fields}
        selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle', "")
        for field in priority_fields:
            if user_input[field] is None:
                user_input[field] = ""

        secondary_muscles = request.form.getlist('secondaryMuscles[]')
        secondary_muscles_str = ' '.join(secondary_muscles)

        user_content = (
            selected_primary_muscle * 20 + ' ' +
            ''.join(map(str, user_input['level'])) * priority_weights[0] + ' ' +
            ''.join(map(str, user_input['equipment'])) * priority_weights[1] + ' ' +
            secondary_muscles_str * priority_weights[2] + ' ' +
            ''.join(map(str, user_input['force'])) * priority_weights[3] + ' ' +
            ''.join(map(str, user_input['mechanic'])) * priority_weights[4] + ' ' +
            ''.join(map(str, user_input['category'])) * priority_weights[5]
        )

        user_tfidf_matrix = tfidf_vectorizer.transform([user_content])
        user_cosine_sim = linear_kernel(user_tfidf_matrix, tfidf_matrix)
        sim_scores = user_cosine_sim[0]
        exercise_indices = sim_scores.argsort()[::-1][:5]

        exercise_ids = [str(df.iloc[index]["id"]) for index in exercise_indices]

        for exercise_id in exercise_ids:
            exercise_doc = collection.find_one({"id": exercise_id})
            if exercise_doc:
                if 'instructions' in exercise_doc:
                    exercise_doc['instructions'] = exercise_doc['instructions'].replace('.,', '<br>')
                exercise_data.append(exercise_doc)

        return render_template('recommendations.html', recommendations=exercise_data, user_input=user_input, selectedPrimaryMuscle=selected_primary_muscle)
    return render_template('recommendations.html', recommendations=exercise_data, user_input=user_input, selectedPrimaryMuscle=selected_primary_muscle)

cosine_sim_items = cosine_similarity(tfidf_matrix.T, tfidf_matrix.T)

@app.route('/more_recommendations', methods=['GET', 'POST'])
def more_recommendations():
    exercise_data = []
    selected_primary_muscle = ""

    if request.method == 'POST':
        selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle', "")

        user_input = json.loads(request.form.get('user_input', '{}'))

        secondary_muscles = request.form.getlist('secondaryMuscles[]')
        secondary_muscles_str = ' '.join(secondary_muscles)

        user_content = (
            selected_primary_muscle * 20 + ' ' +
            ''.join(map(str, user_input.get('level', ''))) * priority_weights[0] + ' ' +
            ''.join(map(str, user_input.get('equipment', ''))) * priority_weights[1] + ' ' +
            secondary_muscles_str * priority_weights[2] + ' ' +
            ''.join(map(str, user_input.get('force', ''))) * priority_weights[3] + ' ' +
            ''.join(map(str, user_input.get('mechanic', ''))) * priority_weights[4] + ' ' +
            ''.join(map(str, user_input.get('category', ''))) * priority_weights[5]
        )

        user_tfidf_matrix = tfidf_vectorizer.transform([user_content])
        user_cosine_sim = cosine_similarity(user_tfidf_matrix, tfidf_matrix)

        item_sim_scores = cosine_similarity(user_cosine_sim, tfidf_matrix.T)[0]
        
        exercise_indices = item_sim_scores.argsort()[-5:][::-1]
        exercise_ids = [str(df.iloc[index]["id"]) for index in exercise_indices]

        for exercise_id in exercise_ids:
            exercise_doc = collection.find_one({"id": exercise_id})
            if exercise_doc:
                if 'instructions' in exercise_doc:
                    exercise_doc['instructions'] = exercise_doc['instructions'].replace('.,', '<br>')
                exercise_data.append(exercise_doc)

    return render_template('more_recommendations.html', recommendations=exercise_data, selectedPrimaryMuscle=selected_primary_muscle)

@app.route('/fitnessCalc')
def fitnessCalc():
    return render_template('fitnessCalc.html')

@app.route('/fatLose')
def fatLose():
    return render_template('weightLoss.html')

@app.route('/muscleGain')
def muscleGain():
    return render_template('muscleGain.html')

if __name__ == '__main__':
    app.run(debug=True)
