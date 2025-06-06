import seaborn as sns


############ COLORS ############
palette = sns.color_palette("Paired")
blue            = palette[1]
light_blue      = palette[0]
green           = palette[3]
light_green     = palette[2]
red             = palette[5]
orange          = palette[7]
light_orange    = palette[6]


############ NAMES ############
NO_CBM = "ConceptBottleneckModelNoConceptSupervisionReLU"
NO_CBM_plus = "ConceptBottleneckModelNoConceptSupervisionReLU_ExtraCapacity"
CBM_bool = "ConceptBottleneckModelBool"
CBM_fuzzy = "ConceptBottleneckModelFuzzy"
CBM_fuzzy_plus = "ConceptBottleneckModelFuzzyExtraCapacity_LogitOnlyExtra"
MaskedBoundedEmbedding = "MaskedSplitEmbModelBounded"
SplitEmbedding = "SplitEmbModelSharedProb_SigmoidalOnlyProb"


name_mappings = {
    NO_CBM: "No concepts",
    NO_CBM_plus: "VanillaModel+",
    CBM_bool: "Bool",
    CBM_fuzzy: "Fuzzy",
    CBM_fuzzy_plus: 'Hybrid',
    MaskedBoundedEmbedding: "Masked Bounded Embedding",
    SplitEmbedding:  'Isolated Concept Embedding (ours)',
}

