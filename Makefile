.PHONY: clean environment requirements updaupdate-environment toy-models models

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = deep-learning-vlae

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Update conda environment
update-environment:
	conda env update -n ${PROJECT_NAME}

## Install conda environment
environment:
	conda env create -n ${PROJECT_NAME}

## Install requirements on hpc
hpc-requirements:
	pip3 install --user -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Train toy models
toy-models: 
	bsub < batch_scripts/train_toy_models.sh 

## Train all models
models: word-models character-models

## Train all word models
word-models: models/WordRAE/finished \
	models/WordVRAE/finished models/WordVRAEIAF/finished

models/Word%/finished: batch_scripts/train_Word%.sh # FORCE
	bsub < batch_scripts/train_Word$*.sh 
	# python3 src/models/word_models.py Word$*

## Train all character models
character-models: models/CharacterRAE/finished \
	models/CharacterVRAE/finished models/CharacterVRAEIAF/finished

models/Character%/finished: batch_scripts/train_Character%.sh 
	bsub < batch_scripts/train_Character$*.sh 
	# python3 src/models/character_models.py Character$*

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
