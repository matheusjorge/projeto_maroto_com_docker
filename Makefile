activate_venv:
	sh source ./venvs/${venv}/bin/activate

activate_docker:
	launchctl start /usr/local/bin/docker
