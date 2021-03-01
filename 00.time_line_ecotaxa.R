library("tidyverse")
library("lubridate")
library("sp")

# library("ecotaxar")
# db <- src_ecotaxa()
# s <- tbl(db, "obj_head") %>%
#   group_by(sampleid) %>%
#   select(objid, objdate, latitude, longitude) %>%
#   slice_min(objid, n=1) %>%
#   # show_query()
#   collect()
# write_csv(s, "data/samples.csv")

w <- read_csv("data/gshhg_world_c.csv")
s <- read_csv("data/samples.csv") %>%
  rename(lon=longitude, lat=latitude, date=objdate)

s <- s %>%
  filter(lon >= -180, lon <=180, lat >= -90, lat <= 90) %>%
  filter(date > "2000-01-01") %>%
  filter(!sp::point.in.polygon(lon, lat, pol.x=w$lon, pol.y=w$lat)) %>%
  mutate(year=year(date))

lims <- c(2008, 2012, 2016, 2020)
sc <- map_dfr(1:length(lims), function(i) {
  s %>%
    filter(year <= lims[i]) %>%
    mutate(period=str_c("-> ", lims[i]))
})


ggplot(mapping=aes(lon, lat)) + coord_quickmap() +
  chroma::scale_xy_map() +
  geom_polygon(data=w, fill="grey50") +
  geom_point(data=sc, size=0.2) +
  facet_wrap(~period)
