from flask import Flask, render_template, request, jsonify
import sys
import os
from importlib import import_module


def load_retriever():
    # Try import via package, fallback to loading from file path
    try:
        mod = import_module('src.inference')
        BISRetriever = getattr(mod, 'BISRetriever')
    except Exception:
        # Fallback: load directly from file
        import importlib.util
        spec = importlib.util.spec_from_file_location('src.inference', os.path.join(os.getcwd(), 'src', 'inference.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BISRetriever = getattr(mod, 'BISRetriever')
    return BISRetriever()


def create_app(testing=False):
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
    app.config['TESTING'] = testing

    retriever = None
    try:
        retriever = load_retriever()
    except Exception as e:
        app.logger.error('Failed to load retriever: %s', e)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/infer', methods=['POST'])
    def infer():
        data = request.get_json() if request.is_json else request.form
        query = data.get('query') if data else None
        top_k = int(data.get('top_k', 5)) if data else 5
        if not query:
            return jsonify({'error': 'query parameter required'}), 400
        if retriever is None:
            return jsonify({'error': 'Retriever failed to initialize on server.'}), 500
        results = retriever.retrieve(query, top_k=top_k)
        return jsonify({'query': query, 'results': results})

    @app.route('/evaluate', methods=['POST'])
    def evaluate():
        data = request.get_json() if request.is_json else request.form
        results_path = data.get('results_path') if data else None
        if not results_path:
            return jsonify({'error': 'results_path required'}), 400
        # Run eval_script.py via subprocess using current python
        import subprocess, sys
        cmd = [sys.executable, os.path.join(os.getcwd(), 'eval_script.py'), '--results', results_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return jsonify({'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/results')
    def results_page():
        q = request.args.get('q', '')
        return render_template('results.html', query=q)

    @app.route('/instructions')
    def instructions_page():
        return render_template('instructions.html')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=5000, debug=True)
