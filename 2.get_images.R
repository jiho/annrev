library("tidyverse")
library("fs")
library("magick")

message("Read data table") #----

f <- read_tsv("data/regent_data.tsv.gz", col_types=cols_only(objid="i", file_name="c"))


message("Copy images from EcoTaxa's vault") #----

orig_dir <- path_expand("~/datasets/regent_ptB/orig/")
dir_create(orig_dir, recurse=TRUE)

vault_imgs <- path("/remote","ecotaxa","vault", f$file_name)
local_imgs <- path(orig_dir, str_c(f$objid, ".jpg"))

exists <- file.exists(local_imgs)
message(" ", sum(!exists), " images to copy")

copied <- file_copy(vault_imgs[!exists], local_imgs[!exists], overwrite=TRUE)


message("Chop and trim images")  #----

cropped_dir <- path_expand("~/datasets/regent_ptB/cropped/")
dir_create(cropped_dir)

crop <- 31

# do not reprocess existing files
cropped_imgs <- local_imgs %>% str_replace("orig", "cropped") %>% str_replace("jpg", "png")
exists <- file_exists(cropped_imgs)
message(" ", sum(!exists), " images to process")

# then go
walk(which(!exists), function(i) {
  img <- image_read(local_imgs[i])
  geom <- str_c("x",image_info(img)$height-crop, "+0+", crop)
  img <- image_crop(img, geometry=geom, gravity="south") %>% image_trim()
  image_write(img, cropped_imgs[i], format="png")
})
