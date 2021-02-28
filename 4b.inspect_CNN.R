#
# Inspect the result of a CNN fit and evaluation
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("tidyverse")
source("lib_ml_utils.R")

# output dir of the CNN training
output_dir <- "/home/jiho/datasets/regent_ptB/full_train/"
taxo_precision <- "detailed"


## Training history ----

# read the history log
h <- read_tsv(file.path(output_dir, "train_history.tsv"))

hl <- h %>%
  pivot_longer(cols=train_loss:val_accuracy, names_to="var", values_to="value") %>%
  separate(var, into=c("dataset", "metric"))

h %>%
  ggplot() +
    geom_path(aes(x=epoch, y=val, colour=dataset)) +
    facet_wrap(~metric, scales="free_y")


## Prediction quality ----

d <- read_tsv(str_c("results/CNN-", taxo_precision, "-predictions.tsv.gz"))

# split into sets
sets <- split(d, d$set)

# evaluate prediction quality on the test set
cm_test <- with(sets$test, cm(true=taxon, pred=cnn_taxon))
plot(cm_test, trans="log1p")
ggsave(str_c("results/CNN-", taxo_precision, "-confusion_matrix.pdf"), width=8, height=7)

cr_test <- with(sets$test, classification_report(true=taxon, pred=cnn_taxon))
(cr_table <- plot(cr_test))
gt::gtsave(cr_table, str_c("results/CNN-", taxo_precision, "-classif_report.html"))
# Detailed
# -> the classes that contaminate most the others are
#      _detritus
#      _mix
#      part<Crustacea
#      _badfocus<artefact
# -> the classes collect most objects from the others are
#      _detritus
#      Dyphidae (to a much lower extent)
