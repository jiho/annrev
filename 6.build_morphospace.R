#
# Construct a morphological space on the regent dataset
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3


library("tidyverse")
# remotes::install_github("jiho/morphr")
library("morphr")
library("bestNormalize")

# read data
d <- read_tsv("data/regent_data.tsv.gz", col_types=cols())

# subsample data
set.seed(1)
ds <- d %>%
  # keep only the living
  filter(str_detect(taxon_detailed, "^_")) %>%
  # subsample data for quicker plot
  sample_n(10000)

# normalise the distribution of each feature

#' Remove the extremes of each variable of a table
#'
#' @param x a data.frame like object
#' @param percent percentage of data to remove at each extreme
#'
#' @return The input data.frame with NA where the extreme were
drop_extreme <- function(x, percent=1) {
  x[x < quantile(x, percent/100, na.rm=TRUE)] <- NA
  x[x > quantile(x, 1 - (percent/100), na.rm=TRUE)] <- NA
  return(x)
}

show_distrib <- function(x, n=1000) {
  x %>%
    sample_n(n) %>% pivot_longer(cols=area:cdexc) %>%
    ggplot() + geom_density(aes(x=value)) + facet_wrap(~name, scale="free")
}

# select features
feat <- ds %>%
  select(area:cdexc)

show_distrib(feat)
feat <- mutate(feat, across(.fns=drop_extreme, percent=1))
show_distrib(feat)
feat <- mutate(feat, across(.fns=function(x) {yeojohnson(x)$x.t}))
show_distrib(feat)

# build morphospace
space <- morpho_space(feat, weights=rep(1, nrow(featn)))
# and plot it
morphr::ggmorph_tile(space, imgs=str_c("~/datasets/regent_ptB/cropped/", ds$objid, ".png"), steps=13, n_imgs=8, scale=0.004)
