#.PHONY: test-training-agent-interface
#test-training-agent-interface:
#	PYTHONHASHSEED=24 pytest -v
#
#.PHONY: test-learning
#test-learning:
#
#.PHONY: profiling
#profiling:
# run visualization for cov_html: ruby -run -ehttpd . -p8000
.PHONY: clean
clean:
	rm -rf ./logs/*

.PHONY: format
format:
	# pip install black==20.8b1
	black --exclude third_party .

.PHONY: rm-pycache
rm-pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: compile
compile:
	python -m grpc_tools.protoc -I uniagent/core/protos --python_out=uniagent/core/service --pyi_out=uniagent/core/service --grpc_python_out=uniagent/core/service uniagent/core/protos/env_server.proto
	python -m grpc_tools.protoc -I uniagent/core/protos --python_out=uniagent/core/service --pyi_out=uniagent/core/service --grpc_python_out=uniagent/core/service uniagent/core/protos/agent_server.proto