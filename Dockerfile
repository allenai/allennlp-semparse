FROM python:3.7.2

# Setup a spot for the code
WORKDIR /allennlp_semparse

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY .flake8 .flake8
COPY pytest.ini pytest.ini
COPY pyproject.toml pyproject.toml
COPY training_config/ training_config/
COPY tests tests/
COPY test_fixtures test_fixtures/
COPY allennlp_semparse allennlp_semparse/

CMD ["/bin/bash"]
