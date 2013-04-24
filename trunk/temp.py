#!/usr/bin/env python

import ystockquote
    
    
temp = ystockquote.get_historical_prices('GOO', '20130410', '20130412')
print temp 
