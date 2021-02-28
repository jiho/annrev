#
# Compare predictions vs truth on time series
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("tidyverse")

# read randomforest predictions
# rf_p <- read_tsv("results/RF-coarse-predictions.tsv.gz", col_types=cols())
rf_p <- read_tsv("results/RF-detailed-predictions.tsv.gz", col_types=cols())
cnn_p <- read_tsv("results/CNN-detailed-predictions.tsv.gz", col_types=cols())

d <- left_join(rf_p, select(cnn_p, objid, cnn_taxon))

# extract lineages
lineages <- distinct(d, taxon, lineage)

#TODO look at lineages below

# put RF vs true predictions in the same column
ds <- d %>%
  select(-lineage) %>%
  rename(true=taxon, RF=rf_taxon, CNN=cnn_taxon) %>%
  pivot_longer(cols=c("true", "RF", "CNN"), names_to="source", values_to="taxon") %>%
  left_join(lineages)

# compute concentration
dc <- ds %>%
  # compute unit concentration
  mutate(unit_conc=1*sub_part/tot_vol) %>%
  # sum per date
  group_by(date, lineage, taxon, source) %>%
  summarise(conc=sum(unit_conc), .groups="drop") %>%
  # add missing zeroes
  complete(date, source, taxon, fill=list(conc=0))

# compute overall abundance per taxon
by_taxon <- dc %>%
  filter(source=="true") %>%
  group_by(taxon) %>%
  summarise(total_conc=sum(conc), .groups="drop") %>%
  arrange(desc(total_conc))
# 20 most abundant
head(by_taxon, 20)
# 20 least abundant
tail(by_taxon, 20)

## Comparison on time series ----

# plot all time series for reference
p <- ggplot(dc, aes(x=date, y=conc, colour=source)) +
  # geom_point(size=0.2) +
  geom_smooth(se=F, span=0.02, n=500, size=0.5) +
  facet_grid(taxon~., scales="free_y", switch="y") +
  scale_y_continuous(trans="log1p") +
  theme(legend.position="top")
ggsave("results/all_time_series.pdf", p, width=8, height=60, dpi=100, limitsize=FALSE)

# Chaetognatha and Phaeodaria have the two best F1 scores, yet look quite different in terms of time series
# focus_taxa <- c("Copepoda", "Harosa", "Chaetognatha")
focus_taxa <- c("Calanidae", "Temoridae", "Chaetognatha", "Phaeodaria", "Echinodermata")

filter(dc, taxon %in% focus_taxa) %>%
  mutate(
    taxon=factor(taxon, levels=focus_taxa),
    source=factor(source, levels=c("true", "RF", "CNN"))
  ) %>%
  ggplot(aes(x=date, y=conc, colour=source, linetype=source)) +
  # geom_path(size=0.2) +
  stat_smooth(geom="line", se=F, span=0.01, n=800, size=0.3, alpha=0.9, lineend="round") +
  facet_grid(taxon~., scales="free_y", switch="y") +
  scale_y_continuous(trans="sqrt") +
  theme_light() +
  theme(legend.position="top") +
  scale_colour_manual(values=c("black", "#eb5539", "#006FCA")) +
  scale_linetype_manual(values=c("21", "solid", "solid")) +
  labs(x="Date", y="Concentration [#/m3]", colour="Source", linetype="Source")
ggsave("results/selected_ts.png", width=8, height=7, dpi=100)
ggsave("results/selected_ts.pdf", width=8, height=7)

filter(dc, taxon %in% focus_taxa) %>%
  select(-lineage) %>%
  mutate(taxon=factor(taxon, levels=focus_taxa)) %>%
  pivot_wider(names_from="source", values_from="conc") %>%
  pivot_longer(CNN:RF, names_to="model", values_to="predicted") %>%
  ggplot() +
  geom_abline(aes(slope=1, intercept=0)) +
  geom_point(aes(x=true, y=predicted, colour=model), size=0.25, alpha=0.5) +
  scale_x_continuous(trans="log10") + scale_y_continuous(trans="log10") +
  scale_colour_manual(values=c("#308dc7", "#eb5539")) +
  # facet_wrap(~taxon) +
  facet_grid(model~taxon) +
  coord_fixed()
ggsave("results/pred_vs_true.png", width=8, height=4, dpi=100)


## Focus on Copepods ----

# get only copepods
cops <- dc %>%
  filter(str_detect(lineage, "Copepoda")) %>%
  select(-lineage)

# convert to wide format
# put true and RF as separate columns
copsw <- cops %>%
  mutate(conc=log1p(conc*100)) %>%
  pivot_wider(names_from=c("taxon", "source"), values_from="conc", values_fill=0) %>%
  relocate(ends_with("_true"), .after="date") %>%
  select(-date)

library("FactoMineR")
# perform PCA putting RF-predicted taxa as supplementary
pca <- PCA(copsw, quanti.sup=which(str_detect(names(copsw), "RF")), graph=FALSE)
plot(pca, choix="var")

