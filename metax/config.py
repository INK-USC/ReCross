METRICS = {
    "super_glue-wsc.fixed": "EM|SoftEM",
    "winogrande-winogrande_xl": "EM|SoftEM",
    "super_glue-cb": "EM|SoftEM",
    "super_glue-rte": "EM|SoftEM",
    "anli": "EM|SoftEM",
    "anli_r1": "EM|SoftEM",
    "anli_r2": "EM|SoftEM",
    "anli_r3": "EM|SoftEM",
    "super_glue-copa": "EM|SoftEM",
    "story_cloze-2016": "EM|SoftEM",
    "hellaswag": "EM|SoftEM",
    "super_glue-wic": "EM|SoftEM",
    "task": "EM|SoftEM"
}

DOWNSTREAM_DATASETS = [
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    "super_glue-rte",
    "anli",
    "super_glue-copa",
    "story_cloze-2016",
    "hellaswag",
    "super_glue-wic"
]

DOWNSTREAM_DATASETS_foldernames = [
"winogrande-winogrande_xl",
"super_glue-cb",
"super_glue-rte",
"anli_r1",
"anli_r2",
"anli_r3",
# "anli",
"story_cloze-2016",
"super_glue-wsc.fixed",
"super_glue-copa",
"hellaswag",
"super_glue-wic",
]


# T0pp's training.

# T0pp's training.
UPSTREAM_DATASETS = [
    "glue-mrpc",
		"glue-qqp",
		"paws_x-en",
		"kilt_tasks-hotpotqa",
		"wiki_qa",
		"adversarial_qa-dbert",
		"adversarial_qa-dbidaf",
		"adversarial_qa-droberta",
		"duorc-SelfRC",
		"duorc-ParaphraseRC",
		"ropes",
		"quoref",
		"cos_e-v1.11",
		"cosmos_qa",
		"dream",
		"qasc",
		"quail",
		"quartz",
		"sciq",
		"social_i_qa",
		"wiki_hop-original",
		"wiqa",
		"amazon_polarity",
		"app_reviews",
		"imdb",
		"rotten_tomatoes",
		"yelp_review_full",
		"common_gen",
		"wiki_bio",
		"cnn_dailymail-3.0.0",
		"gigaword",
		"multi_news",
		"samsum",
		"xsum",
		"ag_news",
		"dbpedia_14",

		"trivia_qa-unfiltered",
		"ai2_arc-ARC-Easy",
		"race-high",
		"piqa",
		"ai2_arc-ARC-Challenge",
		"squad_v2",
		"hellaswag",
		"openbookqa-main",
		"race-middle",
		"web_questions",

		"super_glue-multirc",
		"super_glue-boolq",
		"super_glue-wic",
		"super_glue-copa",
		"super_glue-record",
		"super_glue-wsc.fixed" # note that this is not in the BART0pp's training.
]


CSR_UPSTEREAM_TASKS = [
	"commonsense_qa",
	"social_i_qa",
	"ai2_arc-ARC-Challenge",
	"ai2_arc-ARC-Easy",
	"openbookqa-main",
	"openbookqa-additional",
	"hellaswag",
	# "winograd_wsc-wsc285",
	# "winograd_wsc-wsc273",
	"winogrande-winogrande_xl",
	"super_glue-copa",
	"super_glue-record",
	"codah-codah",
	# "mc_taco",
	"qasc",
	"common_gen",
	"cosmos_qa",
	"dream",
	"piqa",
	"wiqa",
	"art",
	# "numer_sense",
]

# print(set(UPSTREAM_DATASETS) & set(DOWNSTREAM_DATASETS))

FULL_TARGET_TASKS_FOR_T0 = [
    "ai2_arc-ARC-Easy",
    "piqa",
    "ai2_arc-ARC-Challenge",
    "squad_v2",
    "openbookqa-main",
    "super_glue-multirc",
    "super_glue-boolq",
    "super_glue-wic",
    "super_glue-copa",
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    "super_glue-rte",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "hellaswag",
]

CSR_TARGET_TASKS = [
    "squad_v2",
    "super_glue-boolq",
    "super_glue-wic",
    "super_glue-cb",
    "super_glue-rte",
    "anli_r1",
    "anli_r2",
    "anli_r3",
	"glue-mrpc",
	"glue-qqp",
	"rotten_tomatoes",
	"imdb",
	"ag_news"
]
