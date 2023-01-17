A collection of 1-m historical price data. The data is collected from yFinance.

* Occasionally, yFinacne does records a price as NaN. When this is the case,
  I linearly interpolated the price to remove the NaN. It seems that most NaN
  recorded values occured in PM or AH where there is usually not much volume.
  The linearly interpolated price should not affect much in this case.

* Filtered - a median filter was applied to remove the spikes that occasionally
  occur in PM or AH most likely due to a market order that should have been
  placed as a limit order.

* Unfiltered - spikes not removed
