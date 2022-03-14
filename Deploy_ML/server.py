import model # Import the python file containing the ML model
from flask import Flask, request, render_template # Import flask libraries


# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('home.html') # Render home.html


# Route 'classify' accepts GET request
@app.route('/classify',methods=['GET'])
def classify_type():
    try:
        tweet = request.args.get('tweet') # Get parameters for sepal length
       

        # Get the output from the classification model
        tweet_pred = model.classify(tweet)

        # Render the output in new HTML page
        return render_template('output.html', tweet_pred=tweet_pred)
    except:
        return 'Error'    




# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=True)