#include <iostream>
#include <vector>
#include <string>
#include <chrono>    // For timestamps
#include <random>    // For generating random orders
#include <iomanip>   // For formatting output
#include <thread>    // For potential delays
#include <map>       // For storing price levels
#include <list>      // For storing orders at a price level (FIFO)
#include <limits>    // For numeric limits (e.g., infinity)
#include <functional>// For std::greater/std::less
#include <vector>    // Can be used instead of list if preferred
#include <cmath>     // For std::round
#include <stdexcept> // For exceptions
#include <algorithm> // For std::min

// --- Enums for Order Properties ---
enum class OrderSide { BUY, SELL };
enum class OrderType { LIMIT, MARKET };

// --- Forward Declarations ---
// We need to tell the compiler that OrderBook exists before MatchingEngine uses it.
class OrderBook;

// --- Order Structure ---
struct Order {
    long long orderId;
    OrderSide side;
    OrderType type;
    double price;
    int quantity;
    long long timestamp;
    int initialQuantity;

    Order(long long id, OrderSide s, OrderType t, double p, int q)
        : orderId(id), side(s), type(t), price(p), quantity(q), initialQuantity(q) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    }
    Order() : orderId(0), side(OrderSide::BUY), type(OrderType::LIMIT), price(0.0), quantity(0), initialQuantity(0), timestamp(0) {}

    void print(bool showTimestamp = true) const {
        std::cout << "ID:" << orderId
                  << " " << (side == OrderSide::BUY ? "BUY" : "SELL")
                  << " " << (type == OrderType::LIMIT ? "LIM" : "MKT")
                  << " Qty:" << quantity << "/" << initialQuantity;
        if (type == OrderType::LIMIT) {
            std::cout << " Px:" << std::fixed << std::setprecision(2) << price;
        }
        if (showTimestamp) {
             std::cout << " T:" << timestamp;
        }
    }
};

// --- Trade Structure ---
// Represents a completed trade execution
struct Trade {
    long long tradeId;        // Unique ID for the trade itself
    long long aggressingOrderId; // ID of the order that initiated the trade (taker)
    long long restingOrderId;  // ID of the order that was resting in the book (maker)
    double price;             // Price at which the trade occurred
    int quantity;             // Quantity traded
    long long timestamp;      // Time of the trade execution

    // Static counter for generating unique trade IDs
    static long long nextTradeId;

    Trade(long long aggressingId, long long restingId, double p, int q)
        : aggressingOrderId(aggressingId), restingOrderId(restingId), price(p), quantity(q) {
        tradeId = nextTradeId++; // Assign and increment unique ID
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    }

     void print() const {
        std::cout << "TRADE ID:" << tradeId
                  << " Aggressor:" << aggressingOrderId
                  << " Resting:" << restingOrderId
                  << " Qty:" << quantity
                  << " Price:" << std::fixed << std::setprecision(2) << price
                  << " Timestamp:" << timestamp << std::endl;
    }
};

// Initialize static member
long long Trade::nextTradeId = 1;

// --- Order Book Class (definition remains the same as before) ---
class OrderBook {
// ... (Keep the entire OrderBook class implementation from the previous version) ...
// NOTE: Make sure the OrderBook class definition is fully included here.
// For brevity in this display, it's omitted, but it MUST be present in the actual file.
private:
    using OrderList = std::list<Order>;
    std::map<double, OrderList, std::greater<double>> bids;
    std::map<double, OrderList, std::less<double>> asks;
    // TODO: Add efficient cancellation map

public:
    void addOrder(const Order& order) {
        if (order.type != OrderType::LIMIT || order.quantity <= 0 || order.price <= 0) {
             // std::cerr << "Warning: Ignoring invalid or non-LIMIT order ID " << order.orderId << " in addOrder" << std::endl;
            return;
        }
        if (order.side == OrderSide::BUY) {
            bids[order.price].push_back(order);
        } else {
            asks[order.price].push_back(order);
        }
         // std::cout << "Book: Added Order ID " << order.orderId << " to " << (order.side == OrderSide::BUY ? "Bids" : "Asks") << " at price " << order.price << std::endl;
    }

    // IMPORTANT: This needs modification for the matching engine
    // It should remove the *specific* order object, not just based on ID search
    // We will modify resting orders directly via iterators/references instead.
    // This basic ID-based removal is less useful now.
    bool removeOrder(long long orderId) {
        // Inefficient Search - Keep for now, but matching engine won't use it this way
        for (auto& pair : bids) {
            auto& orderList = pair.second;
            for (auto it = orderList.begin(); it != orderList.end(); ++it) {
                if (it->orderId == orderId) {
                    orderList.erase(it);
                    if (orderList.empty()) bids.erase(pair.first);
                    return true;
                }
            }
        }
        for (auto& pair : asks) {
            auto& orderList = pair.second;
            for (auto it = orderList.begin(); it != orderList.end(); ++it) {
                if (it->orderId == orderId) {
                    orderList.erase(it);
                    if (orderList.empty()) asks.erase(pair.first);
                    return true;
                }
            }
        }
        return false;
    }

    double getBestBid() const {
        return bids.empty() ? 0.0 : bids.begin()->first;
    }

    double getBestAsk() const {
        return asks.empty() ? std::numeric_limits<double>::infinity() : asks.begin()->first;
    }

    bool bidsEmpty() const { return bids.empty(); }
    bool asksEmpty() const { return asks.empty(); }

    // Return reference to the list of orders at the best price
    // WARNING: Check !empty() before calling
    OrderList& getBestBidOrders() {
        if (bids.empty()) throw std::runtime_error("Bids empty");
        return bids.begin()->second;
    }
    OrderList& getBestAskOrders() {
        if (asks.empty()) throw std::runtime_error("Asks empty");
        return asks.begin()->second;
    }

    // Removes the price level if its order list becomes empty
    void cleanEmptyPriceLevel(OrderSide side, double price) {
        if (side == OrderSide::BUY) {
            auto it = bids.find(price);
            if (it != bids.end() && it->second.empty()) {
                bids.erase(it);
                // std::cout << "Cleaned empty bid price level: " << price << std::endl;
            }
        } else { // SELL
            auto it = asks.find(price);
            if (it != asks.end() && it->second.empty()) {
                asks.erase(it);
                // std::cout << "Cleaned empty ask price level: " << price << std::endl;
            }
        }
    }

    void printBook() const {
         std::cout << "\n--- Order Book State ---" << std::endl;
         std::cout << "ASKS (Price | Total Qty):" << std::endl;
         if (asks.empty()) { std::cout << "  <Empty>" << std::endl; }
         else {
             std::vector<double> askPrices;
             for(const auto& pair : asks) askPrices.push_back(pair.first);
             for (auto it = askPrices.rbegin(); it != askPrices.rend(); ++it) {
                 double price = *it;
                 const auto& orderList = asks.at(price);
                 int totalQuantity = 0;
                 for(const auto& order : orderList) totalQuantity += order.quantity;
                 std::cout << "  " << std::fixed << std::setprecision(2) << price << " | " << totalQuantity << " (";
                 for (const auto& order : orderList) { std::cout << order.orderId << ":" << order.quantity << " "; }
                 std::cout << ")" << std::endl;
             }
         }
         std::cout << "------------------------" << std::endl;
         std::cout << "BIDS (Price | Total Qty):" << std::endl;
         if (bids.empty()) { std::cout << "  <Empty>" << std::endl; }
         else {
             for (const auto& pair : bids) {
                 double price = pair.first;
                 const auto& orderList = pair.second;
                 int totalQuantity = 0;
                 for(const auto& order : orderList) totalQuantity += order.quantity;
                 std::cout << "  " << std::fixed << std::setprecision(2) << price << " | " << totalQuantity << " (";
                 for (const auto& order : orderList) { std::cout << order.orderId << ":" << order.quantity << " "; }
                 std::cout << ")" << std::endl;
             }
         }
         std::cout << "--- End Order Book ---" << std::endl;
    }
};


// --- Matching Engine Class ---

class MatchingEngine {
private:
    OrderBook& book; // Reference to the order book
    std::vector<Trade>& trades; // Reference to a vector to store generated trades

public:
    // Constructor: Takes references to the OrderBook and the trade list
    MatchingEngine(OrderBook& ob, std::vector<Trade>& tradeList)
        : book(ob), trades(tradeList) {}

    // Processes an incoming order
    void processOrder(Order& incomingOrder) { // Pass by reference to modify quantity
        std::cout << "Engine Processing: ";
        incomingOrder.print(false); // Print without timestamp for brevity
        std::cout << std::endl;

        if (incomingOrder.quantity <= 0) {
            std::cout << "  Ignoring order with zero or negative quantity." << std::endl;
            return;
        }

        if (incomingOrder.type == OrderType::LIMIT) {
            processLimitOrder(incomingOrder);
        } else { // OrderType::MARKET
            processMarketOrder(incomingOrder);
        }
    }

private:
    // Processes a LIMIT order
    void processLimitOrder(Order& incomingOrder) {
        bool matched = false;
        if (incomingOrder.side == OrderSide::BUY) {
            // Try to match against existing asks
            matched = matchBuyOrder(incomingOrder);
        } else { // OrderSide::SELL
            // Try to match against existing bids
            matched = matchSellOrder(incomingOrder);
        }

        // If the limit order was not fully filled by matching, add the remainder to the book
        if (incomingOrder.quantity > 0) {
            std::cout << "  Adding remaining Qty " << incomingOrder.quantity << " of LIMIT order " << incomingOrder.orderId << " to book." << std::endl;
            book.addOrder(incomingOrder);
        } else {
             std::cout << "  LIMIT order " << incomingOrder.orderId << " fully filled." << std::endl;
        }
    }

    // Processes a MARKET order
    void processMarketOrder(Order& incomingOrder) {
        bool matched = false;
        if (incomingOrder.side == OrderSide::BUY) {
            // Match against existing asks
            matched = matchBuyOrder(incomingOrder);
        } else { // OrderSide::SELL
            // Match against existing bids
            matched = matchSellOrder(incomingOrder);
        }

        if (incomingOrder.quantity > 0) {
            // Market orders that aren't fully filled are typically cancelled
            // (or handled according to specific exchange rules, e.g., Fill-or-Kill)
            std::cout << "  Market order " << incomingOrder.orderId << " partially filled. Remaining Qty "
                      << incomingOrder.quantity << " cancelled (insufficient liquidity)." << std::endl;
            // No action needed to cancel, it just wasn't added to the book.
        } else {
             std::cout << "  Market order " << incomingOrder.orderId << " fully filled." << std::endl;
        }
    }

    // Matches an incoming BUY order (Limit or Market) against resting SELL orders (Asks)
    // Returns true if any part of the order was matched.
    bool matchBuyOrder(Order& incomingBuyOrder) {
        bool matched = false;
        // Keep matching as long as the incoming order has quantity and there are asks
        // AND (for limit orders) the incoming price is >= the best ask price
        // OR (for market orders) there are simply asks available.
        while (incomingBuyOrder.quantity > 0 && !book.asksEmpty() &&
               (incomingBuyOrder.type == OrderType::MARKET || incomingBuyOrder.price >= book.getBestAsk()))
        {
            double bestAskPrice = book.getBestAsk();
            OrderList& bestAskOrders = book.getBestAskOrders(); // Get reference to the list at best price

            // Iterate through orders at the best ask price level (FIFO)
            auto it = bestAskOrders.begin();
            while (it != bestAskOrders.end() && incomingBuyOrder.quantity > 0) {
                Order& restingSellOrder = *it; // Get reference to the resting order

                // Determine trade quantity and price
                int tradeQuantity = std::min(incomingBuyOrder.quantity, restingSellOrder.quantity);
                double tradePrice = restingSellOrder.price; // Trade happens at the resting order's price

                // Generate and store the trade record
                Trade trade(incomingBuyOrder.orderId, restingSellOrder.orderId, tradePrice, tradeQuantity);
                trades.push_back(trade);
                std::cout << "  -> TRADE! ";
                trade.print(); // Print trade details immediately
                matched = true;

                // Update quantities
                incomingBuyOrder.quantity -= tradeQuantity;
                restingSellOrder.quantity -= tradeQuantity;

                // If resting order is filled, remove it from the list
                if (restingSellOrder.quantity <= 0) {
                     std::cout << "  Resting order " << restingSellOrder.orderId << " filled and removed." << std::endl;
                    it = bestAskOrders.erase(it); // erase returns iterator to the next element
                    // TODO: If using efficient cancellation, remove from orderIdToIteratorMap here
                } else {
                    // Resting order partially filled, move to the next resting order in the list
                    ++it;
                }
            } // End of loop through orders at this price level

            // If the list at the best ask price is now empty, remove the price level from the book
            if (bestAskOrders.empty()) {
                book.cleanEmptyPriceLevel(OrderSide::SELL, bestAskPrice);
            }

             // If it was a limit order and it crossed the spread, but couldn't fill more at the next level
             if (incomingBuyOrder.type == OrderType::LIMIT && incomingBuyOrder.quantity > 0 && !book.asksEmpty() && incomingBuyOrder.price < book.getBestAsk()) {
                 break; // Stop matching this limit order
             }

        } // End of while loop (matching across price levels)
        return matched;
    }


    // Matches an incoming SELL order (Limit or Market) against resting BUY orders (Bids)
    // Returns true if any part of the order was matched.
    bool matchSellOrder(Order& incomingSellOrder) {
         bool matched = false;
        // Keep matching as long as the incoming order has quantity and there are bids
        // AND (for limit orders) the incoming price is <= the best bid price
        // OR (for market orders) there are simply bids available.
        while (incomingSellOrder.quantity > 0 && !book.bidsEmpty() &&
               (incomingSellOrder.type == OrderType::MARKET || incomingSellOrder.price <= book.getBestBid()))
        {
            double bestBidPrice = book.getBestBid();
            OrderList& bestBidOrders = book.getBestBidOrders(); // Get reference to the list at best price

            // Iterate through orders at the best bid price level (FIFO)
            auto it = bestBidOrders.begin();
            while (it != bestBidOrders.end() && incomingSellOrder.quantity > 0) {
                Order& restingBuyOrder = *it; // Get reference to the resting order

                // Determine trade quantity and price
                int tradeQuantity = std::min(incomingSellOrder.quantity, restingBuyOrder.quantity);
                double tradePrice = restingBuyOrder.price; // Trade happens at the resting order's price

                // Generate and store the trade record
                // Note: incomingSellOrder is the aggressor here
                Trade trade(incomingSellOrder.orderId, restingBuyOrder.orderId, tradePrice, tradeQuantity);
                trades.push_back(trade);
                 std::cout << "  -> TRADE! ";
                 trade.print(); // Print trade details immediately
                 matched = true;

                // Update quantities
                incomingSellOrder.quantity -= tradeQuantity;
                restingBuyOrder.quantity -= tradeQuantity;

                // If resting order is filled, remove it from the list
                if (restingBuyOrder.quantity <= 0) {
                     std::cout << "  Resting order " << restingBuyOrder.orderId << " filled and removed." << std::endl;
                    it = bestBidOrders.erase(it); // erase returns iterator to the next element
                    // TODO: If using efficient cancellation, remove from orderIdToIteratorMap here
                } else {
                    // Resting order partially filled, move to the next resting order in the list
                    ++it;
                }
            } // End of loop through orders at this price level

            // If the list at the best bid price is now empty, remove the price level from the book
            if (bestBidOrders.empty()) {
                book.cleanEmptyPriceLevel(OrderSide::BUY, bestBidPrice);
            }

             // If it was a limit order and it crossed the spread, but couldn't fill more at the next level
             if (incomingSellOrder.type == OrderType::LIMIT && incomingSellOrder.quantity > 0 && !book.bidsEmpty() && incomingSellOrder.price > book.getBestBid()) {
                 break; // Stop matching this limit order
             }

        } // End of while loop (matching across price levels)
        return matched;
    }

};


// --- Order Generator Function (remains the same) ---
std::vector<Order> generateSampleOrders(int numberOfOrders) {
    std::vector<Order> orders;
    // Use a fixed seed for reproducible results during testing
    // std::mt19937 rng(12345);
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> sideDist(0, 1);
    std::uniform_int_distribution<int> typeDist(0, 4); // ~20% market
    std::uniform_real_distribution<double> priceDist(99.50, 100.50);
    std::uniform_int_distribution<int> qtyDist(10, 50);
    long long currentOrderId = 1000;

    orders.reserve(numberOfOrders);
    for (int i = 0; i < numberOfOrders; ++i) {
        OrderSide side = (sideDist(rng) == 0) ? OrderSide::BUY : OrderSide::SELL;
        OrderType type = (typeDist(rng) == 0) ? OrderType::MARKET : OrderType::LIMIT;
        double price = (type == OrderType::LIMIT) ? round(priceDist(rng) * 100.0) / 100.0 : 0.0;
        int quantity = qtyDist(rng);
        orders.emplace_back(currentOrderId++, side, type, price, quantity);
        // std::this_thread::sleep_for(std::chrono::microseconds(10)); // Can add slight delay
    }
    return orders;
}

// --- Main Function (Updated with Matching Engine) ---

int main() {
    OrderBook book;                 // The order book
    std::vector<Trade> tradeLog;    // A log to store all executed trades
    MatchingEngine engine(book, tradeLog); // The matching engine

    std::cout << "--- Generating Sample Orders ---" << std::endl;
    int numOrdersToGenerate = 25; // Generate a few more orders
    std::vector<Order> incomingOrders = generateSampleOrders(numOrdersToGenerate);

    std::cout << "\n--- Processing Orders with Matching Engine ---" << std::endl;

    // Process generated orders one by one through the matching engine
    for (Order& order : incomingOrders) { // Pass by reference if engine modifies it
         // Print initial book state before processing this order (optional, can be verbose)
         // book.printBook();

        engine.processOrder(order); // Engine handles matching and adding to book if needed

         // Print book state after processing this order (optional)
         // book.printBook();
         // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Slow down for visualization
         std::cout << "---------------------------------------------" << std::endl; // Separator
    }

    std::cout << "\n--- Final Order Book State ---" << std::endl;
    book.printBook();

    std::cout << "\n--- Trade Log (" << tradeLog.size() << " trades) ---" << std::endl;
    if (tradeLog.empty()) {
        std::cout << "<No trades executed>" << std::endl;
    } else {
        for (const auto& trade : tradeLog) {
            trade.print();
        }
    }

    std::cout << "\n--- Finished ---" << std::endl;

    return 0;
}
