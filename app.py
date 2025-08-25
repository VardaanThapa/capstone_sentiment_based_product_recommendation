from flask import Flask, Blueprint, render_template, request
from model import Model

blueprint = Blueprint('main', __name__, template_folder='templates')

@blueprint.route('/', methods=['GET', 'POST'])
def home():
  error = None
  top_products = []

  if request.method == 'POST':
    username = request.form['username']
    error = Model.check_username(username)

    if error is None:
      recommended_products = Model.top_20_recommended_products(username)
      if recommended_products is None:
        error = "No products can be recommended for {}".format(username)
      else:
        top_products = Model.top_5_products(recommended_products)
        print("Final 5 fine-tuned recommended products = {}".format(top_products))
  
  return render_template('index.html', 
            title = "Sentiment-based Recommendation System", form_data = request.form, 
            error = error, top_products = top_products)

def create_app():
  app = Flask(__name__)
  app.register_blueprint(blueprint)
  return app

app = create_app()

if __name__ == "__main__":
  app.run(port=5002)