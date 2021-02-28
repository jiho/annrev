#' Build a confusion matrix
#'
#' @param true vector of true classes.
#' @param pred vector of predicted classes.
confusion_matrix <- function(true, pred) {
  cm <- table(true=true, pred=pred)
  class(cm) <- c("cm", class(cm))
  return(cm)
}
# short cut
cm <- confusion_matrix

#' Plot a confusion matrix
#'
#' @param x a confusion matrix built by confusion_matrix().
#' @param trans transformation function for the color scale.
plot.cm <- function(x, trans="log1p") {
  library("ggplot2")
  as.data.frame(x) %>%
    ggplot() + geom_raster(aes(x=pred, y=true, fill=Freq)) +
    scale_fill_viridis_c(trans=trans) + coord_fixed() +
    theme(axis.text.x=element_text(angle=65, hjust=1))
}

#' Build a classification report
#'
#' @param true vector of true classes.
#' @param pred vector of predicted classes.
classification_report <- function(true, pred) {
  cm <- confusion_matrix(true, pred) %>% as.matrix()

  # basic stats
  n <- sum(cm) # number of instances
  nc <- nrow(cm) # number of classes
  diag <- diag(cm) # number of correctly classified instances per class
  rowsums <- apply(cm, 1, sum) # number of instances per class
  colsums <- apply(cm, 2, sum) # number of predictions per class
  # p <- rowsums / n # distribution of instances over the actual classes
  # q <- colsums / n # distribution of instances over the predicted classes

  # metrics
  accuracy <- sum(diag) / n

  precision <- diag / colsums
  recall <- diag / rowsums
  f1 <- 2 * precision * recall / (precision + recall)

  # classification report
  cr <- data.frame(
    n=table(true),
    precision,
    recall,
    f1
  )
  names(cr)[1:2] <- c("class", "n")
  row.names(cr) <- NULL

  # add global stats
  cr <- bind_rows(
    data.frame(class="accuracy", n=NA, precision=accuracy, recall=accuracy, f1=accuracy),
    data.frame(class="avg", t(apply(cr[,-(1:2)], 2, mean))),
    cr
  )
  class(cr) <- c("cr", class(cr))
  return(cr)
}

show.cr <- function(object) {
  library("gt")
  library("chroma")
  object %>%
    gt() %>%
    data_color(columns=vars(precision, recall, f1), colors=brewer_scale(name="Spectral")) %>%
    fmt_percent(columns=vars(precision, recall, f1), decimals=0, incl_space=TRUE) %>%
    tab_row_group(group="global", rows=1:2)
}
plot.cr <- show.cr
