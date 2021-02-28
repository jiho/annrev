#
# Predict classes with Random Forest
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("tidyverse")
library("ranger")
source("lib_ml_utils.R")
dir.create("results", showWarnings=FALSE)

message("Prepare data") #----

# read data
d <- read_tsv("data/regent_data.tsv.gz", col_types=cols())


# select level of taxonomic precision
taxo_precision <- "detailed"
# taxo_precision <- "coarse"

# replace taxon by the given precision
d$taxon <- d[[str_c("taxon_", taxo_precision)]]

# recompute associated lineage
#' Function that computes string intersection
str_intersect <- function(x) {
  if (length(x) == 1) {
    out <- x
  } else {
    # reduce to shortest string
    min_l <- min(str_length(x))
    # split per character and deduce those which are equal
    common <- str_split_fixed(x, "", n=min_l) %>% apply(2, function(x) {length(unique(x)) == 1})
    # extract the common part of the strings, if any
    if (sum(common) >= 1) {
      out <- str_sub(x[1], 1, max(which(common)))
    } else {
      out <- ""
    }
  }
  return(out)
}
# apply it to the lineages
taxon_lineages <- distinct(d, taxon, lineage) %>%
  group_by(taxon) %>%
  summarise(lineage=str_intersect(lineage)) %>%
  ungroup()
d <- d %>%
  # remove the original lineage
  select(-lineage) %>%
  # add the newly computed one
  left_join(taxon_lineages)


# extract data relevant for model
dm <- select(d, set, taxon, area:cdexc) %>%
  mutate(taxon=factor(taxon)) %>%
  rename(percent_area=`%area`, circ=circ.)

# class weights: square root of ~ inverse frequency
counts <- table(dm$taxon)
weights <- sqrt(max(counts) / counts)

sets <- split(dm, dm$set)

# verify that all taxa are in all data sets
tax <- lapply(sets, function(x) {unique(x$taxon)%>% sort()})
setdiff(tax[[1]], tax[[2]])
setdiff(tax[[1]], tax[[3]])

message("Fit and predict model") #----

# fit model on training set
m <- ranger(
  taxon~.,
  data=sets$train,
  num.trees=300, min.node.size=2,
  class.weights=weights,
  num.threads=20
)

# # check for overfitting on validation set
# val_acc <- map_dfr(seq(1, 300, by=10), function(n) {
#   rf_taxon <- predict(m, data=sets$val, num.trees=n, num.threads=20)$predictions
#   accuracy <- sum(rf_taxon == sets$val$taxon) / nrow(sets$val)
#   tibble(n_trees=n, accuracy)
# })
# ggplot(val_acc) + geom_path(aes(x=n_trees, y=accuracy))
# # -> no overfitting => good

# predict on whole data set
d$rf_taxon <- predict(m, data=dm, num.threads=20)$predictions
# and re-split the dataset
sets <- split(d, dm$set)

message("Inspect results") #----

# for everything
cm(true=d$taxon, pred=d$rf_taxon) %>% plot(trans="log1p")
classification_report(true=d$taxon, pred=d$rf_taxon) %>% plot()

# for the test set
cm_test <- with(sets$test, cm(true=taxon, pred=rf_taxon))
plot(cm_test, trans="sqrt")
ggsave(str_c("results/RF-", taxo_precision, "-confusion_matrix.pdf"), width=8, height=7)
cr_test <- with(sets$test, classification_report(true=taxon, pred=rf_taxon))
cr_table <-plot(cr_test)
gt::gtsave(cr_table, str_c("results/RF-", taxo_precision, "-classif_report.html"))
# Detailed
# -> the classes that contaminate most the others are
#      _detritus
#      _mix
#      Eumalacostraca
#      Dyphidae
# -> the classes collect most objects from the others are
#      _detritus
#      Dyphidae
#      Eumalacostraca
#      part<Crustacea
#      _mix

# store predictions and other metadata to exploit the predictions
select(d, objid, set, sample_name, date, tot_vol, sub_part, lineage, taxon, rf_taxon) %>%
  write_tsv(str_c("results/RF-", taxo_precision, "-predictions.tsv.gz"))
