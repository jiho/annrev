#
# Extract information from the database
#
# (c) 2021 Jean-Olivier Irisson, GNU General Public License v3

library("ecotaxar")
library("tidyverse")

db <- db_connect_ecotaxa()

message("Get data from database") #----

# list Régent projects
projs <- tbl(db, "projects") %>%
  filter(str_detect(title, "Zooscan point B Regent")) %>%
  # subsets for other people
  filter(!str_detect(title, "Siphonophorae")) %>%
  filter(!str_detect(title, "copepodites")) %>%
  # 2020, 2021 not ready
  filter(!str_detect(title, "2020")) %>%
  filter(!str_detect(title, "2021")) %>%
  # clean up
  arrange(title) %>%
  select(projid, title, starts_with("mapping")) %>%
  collect()

projs$title
# -> good, all there

mappings <- distinct(projs, mappingsample, mappingacq, mappingobj)
nrow(mappings)
# -> good, only one

mappings$mappingsample
# t15=tot_vol
mappings$mappingacq
# t03=sub_part
mappings$mappingobj

o <- tbl(db, "objects") %>%
  filter(projid %in% !!projs$projid) %>%
  map_names(mappings) %>%
  left_join(tbl(db, "samples") %>% select(sampleid, tot_vol=t15, sample_name=orig_id)) %>%
  left_join(tbl(db, "acquisitions") %>% select(acquisid, sub_part=t03)) %>%
  left_join(tbl(db, "images") %>% select(objid, file_name)) %>%
  select(projid, sampleid, objid, sample_name, file_name, classif_id, date=objdate, tot_vol, sub_part, area:cdexc) %>%
  collect()


message("Post process data") #----

# add taxonomy
taxo <- extract_taxo(db, o$classif_id)
d <- o %>% mutate(
    tot_vol=as.numeric(tot_vol),
    sub_part=as.numeric(sub_part),
    taxon=taxo_name(classif_id, taxo, unique=TRUE),
    lineage=lineage(classif_id, taxo),
  ) %>%
  select(-classif_id) %>%
  arrange(date)

# split train and test
d <- d %>%
  mutate(year=lubridate::year(d$date)) %>%
  mutate(set=case_when(
    year %in% c(1995, 2019) ~ "train",
    year %in% c(1996, 2018) ~ "val",
    TRUE ~ "test"
  ))

# remap taxonomy into consistent groups
count(d, lineage, taxon, set) %>%
  pivot_wider(names_from="set", values_from="n") %>%
  write_tsv("orig_taxo_mapping.tsv")

taxo_mapping <- read_tsv("taxo_mapping - orig_taxo_mapping.tsv", col_types=cols()) %>%
  select(taxon:taxon_coarse) %>%
  select(starts_with("taxon"))

d <- left_join(d, taxo_mapping)

# remove irrelevant columns
d <- select(d, -x, -y, -bx, -by, -tag)

# remove 0 variance columns
zero_variance <- names(which(select(d, area:cdexc) %>% sapply(var) == 0))
d <- select(d,-all_of(zero_variance))


message("Write data to disk") #----

write_tsv(d, "data/regent_data.tsv.gz")
