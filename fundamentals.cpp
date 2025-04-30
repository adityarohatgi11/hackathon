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

// --- Enums for Order Properties ---

// Represents the side of the order (Buy or Sell)
enum class OrderSide {
    BUY,
    SELL
};

// Represents the type of order (Limit or Market)
// Note: The OrderBook itself primarily deals with LIMIT orders.
// The MatchingEngine handles how MARKET orders interact with the book.
enum class OrderType {
    LIMIT,
    MARKET
};

// --- Order Structure ---

// Represents a single trading order
struct Order {
    long long orderId;        // Unique identifier for the order
    OrderSide side;           // Buy or Sell
    OrderType type;           // Limit or Market
    double price;             // Price for Limit orders (relevant for the book)
    int quantity;             // Number of shares/contracts
    long long timestamp;      // Time the order was created (milliseconds since epoch)
    int initialQuantity;      // Store the original quantity for tracking

    // Constructor
    Order(long long id, OrderSide s, OrderType t, double p, int q)
        : orderId(id), side(s), type(t), price(p), quantity(q), initialQuantity(q) {
        // Get current time as timestamp
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    }

    // Default constructor (needed for some container operations)
    Order() : orderId(0), side(OrderSide::BUY), type(OrderType::LIMIT), price(0.0), quantity(0), initialQuantity(0), timestamp(0) {}


    // Helper function to print order details
    void print() const {
        std::cout << "  Order ID: " << orderId
                  << ", Side: " << (side == OrderSide::BUY ? "BUY" : "SELL")
                  << ", Type: " << (type == OrderType::LIMIT ? "LIMIT" : "MARKET")
                  << ", Qty: " << quantity << "/" << initialQuantity; // Show current/initial quantity
        if (type == OrderType::LIMIT) {
            std::cout << ", Price: " << std::fixed << std::setprecision(2) << price;
        }
        std::cout << ", Timestamp: " << timestamp;
        // Don't add endl here, let the calling print function handle lines
    }
};

// --- Order Book Class ---

class OrderBook {
private:
    // Using std::list to store orders at each price level.
    // std::list provides O(1) insertion/deletion at ends (good for FIFO)
    // and O(1) deletion in the middle if you have an iterator (useful for cancellations).
    using OrderList = std::list<Order>;

    // Bids map: Stores buy orders. Key = Price, Value = List of orders at that price.
    // Uses std::greater<double> to sort prices from highest to lowest.
    std::map<double, OrderList, std::greater<double>> bids;

    // Asks map: Stores sell orders. Key = Price, Value = List of orders at that price.
    // Uses std::less<double> (default) to sort prices from lowest to highest.
    std::map<double, OrderList, std::less<double>> asks;

    // --- TODO: For efficient cancellation (optional enhancement) ---
    // Add an unordered_map to store iterators to orders by orderId
    // std::unordered_map<long long, OrderList::iterator> orderIdToIteratorMap;
    // Remember to also store which map (bids/asks) the iterator belongs to.

public:
    // Adds a LIMIT order to the order book.
    // Market orders are typically handled directly by the MatchingEngine,
    // but a basic book might just reject them or require the engine to handle them.
    void addOrder(const Order& order) {
        // Only add LIMIT orders to the book
        if (order.type != OrderType::LIMIT || order.quantity <= 0 || order.price <= 0) {
             std::cerr << "Warning: Ignoring invalid or non-LIMIT order ID " << order.orderId << std::endl;
            return; // Ignore market orders or orders with invalid details for the book
        }

        if (order.side == OrderSide::BUY) {
            // Add to bids
            bids[order.price].push_back(order);
            // TODO (Cancellation): Store iterator in orderIdToIteratorMap
            // auto it = std::prev(bids[order.price].end());
            // orderIdToIteratorMap[order.orderId] = it; // Need more info (which map)
        } else { // OrderSide::SELL
            // Add to asks
            asks[order.price].push_back(order);
            // TODO (Cancellation): Store iterator in orderIdToIteratorMap
            // auto it = std::prev(asks[order.price].end());
            // orderIdToIteratorMap[order.orderId] = it; // Need more info (which map)
        }
         std::cout << "Added Order ID " << order.orderId << " to " << (order.side == OrderSide::BUY ? "Bids" : "Asks") << " at price " << order.price << std::endl;
    }

    // Removes an order from the book using its ID.
    // This requires the `orderIdToIteratorMap` for efficiency.
    // Basic version (inefficient - O(N) search) provided for structure.
    bool removeOrder(long long orderId) {
        // --- Inefficient Search (demonstration only) ---
        // Search bids
        for (auto& pair : bids) {
            auto& orderList = pair.second;
            for (auto it = orderList.begin(); it != orderList.end(); ++it) {
                if (it->orderId == orderId) {
                    std::cout << "Removing Order ID " << orderId << " from Bids at price " << it->price << std::endl;
                    orderList.erase(it);
                    // If the list at this price level becomes empty, remove the price level
                    if (orderList.empty()) {
                        bids.erase(pair.first);
                    }
                    return true;
                }
            }
        }
        // Search asks
        for (auto& pair : asks) {
            auto& orderList = pair.second;
            for (auto it = orderList.begin(); it != orderList.end(); ++it) {
                if (it->orderId == orderId) {
                     std::cout << "Removing Order ID " << orderId << " from Asks at price " << it->price << std::endl;
                    orderList.erase(it);
                    // If the list at this price level becomes empty, remove the price level
                    if (orderList.empty()) {
                        asks.erase(pair.first);
                    }
                    return true;
                }
            }
        }
         std::cerr << "Warning: Could not find Order ID " << orderId << " to remove." << std::endl;
        return false; // Order not found

        // --- Efficient Version (using orderIdToIteratorMap - TODO) ---
        /*
        auto mapEntry = orderIdToIteratorMap.find(orderId);
        if (mapEntry == orderIdToIteratorMap.end()) {
            return false; // Order not found
        }
        // Need logic here to know whether mapEntry->second belongs to bids or asks
        // Erase using the iterator, then remove from orderIdToIteratorMap
        // ...
        orderIdToIteratorMap.erase(mapEntry);
        return true;
        */
    }


    // Get the highest bid price (Top of Book Bid)
    double getBestBid() const {
        if (bids.empty()) {
            return 0.0; // Or std::numeric_limits<double>::quiet_NaN() or throw exception
        }
        // begin() gives the iterator to the element with the highest price because of std::greater
        return bids.begin()->first;
    }

    // Get the lowest ask price (Top of Book Ask)
    double getBestAsk() const {
        if (asks.empty()) {
            // Return a very high value to indicate no ask, or NaN/exception
            return std::numeric_limits<double>::infinity();
        }
        // begin() gives the iterator to the element with the lowest price because of std::less (default)
        return asks.begin()->first;
    }

    // Check if the order book's bid side is empty
    bool bidsEmpty() const {
        return bids.empty();
    }

    // Check if the order book's ask side is empty
    bool asksEmpty() const {
        return asks.empty();
    }

    // Get a reference to the list of orders at the best bid price level
    // WARNING: Calling code must check bidsEmpty() first!
    OrderList& getBestBidOrders() {
        if (bids.empty()) {
             throw std::runtime_error("Attempted to get best bid orders when bids side is empty.");
        }
        return bids.begin()->second; // Returns reference to the list
    }

    // Get a reference to the list of orders at the best ask price level
    // WARNING: Calling code must check asksEmpty() first!
    OrderList& getBestAskOrders() {
         if (asks.empty()) {
             throw std::runtime_error("Attempted to get best ask orders when asks side is empty.");
         }
        return asks.begin()->second; // Returns reference to the list
    }

     // Removes the price level if its order list becomes empty
    void cleanEmptyPriceLevel(OrderSide side, double price) {
        if (side == OrderSide::BUY) {
            auto it = bids.find(price);
            if (it != bids.end() && it->second.empty()) {
                bids.erase(it);
                 std::cout << "Cleaned empty bid price level: " << price << std::endl;
            }
        } else { // SELL
            auto it = asks.find(price);
            if (it != asks.end() && it->second.empty()) {
                asks.erase(it);
                 std::cout << "Cleaned empty ask price level: " << price << std::endl;
            }
        }
    }


    // Prints the current state of the order book (for debugging)
    void printBook() const {
        std::cout << "\n--- Order Book State ---" << std::endl;

        // Print Asks (sorted lowest to highest price)
        std::cout << "ASKS (Price | Total Qty):" << std::endl;
        if (asks.empty()) {
            std::cout << "  <Empty>" << std::endl;
        } else {
            // Iterate in reverse order of the map keys to print highest asks first visually
             std::vector<double> askPrices;
             for(const auto& pair : asks) {
                 askPrices.push_back(pair.first);
             }
             // Print from highest price down for visual clarity
             for (auto it = askPrices.rbegin(); it != askPrices.rend(); ++it) {
                 double price = *it;
                 const auto& orderList = asks.at(price);
                 int totalQuantity = 0;
                 for(const auto& order : orderList) {
                     totalQuantity += order.quantity;
                 }
                 std::cout << "  " << std::fixed << std::setprecision(2) << price << " | " << totalQuantity << " (";
                 // Optionally print individual order IDs at this level
                 for (const auto& order : orderList) { std::cout << order.orderId << ":" << order.quantity << " "; }
                 std::cout << ")" << std::endl;
             }
        }

        std::cout << "------------------------" << std::endl; // Spread separator

        // Print Bids (sorted highest to lowest price)
        std::cout << "BIDS (Price | Total Qty):" << std::endl;
        if (bids.empty()) {
            std::cout << "  <Empty>" << std::endl;
        } else {
            for (const auto& pair : bids) {
                double price = pair.first;
                const auto& orderList = pair.second;
                int totalQuantity = 0;
                for(const auto& order : orderList) {
                    totalQuantity += order.quantity;
                }
                std::cout << "  " << std::fixed << std::setprecision(2) << price << " | " << totalQuantity << " (";
                 // Optionally print individual order IDs at this level
                 for (const auto& order : orderList) { std::cout << order.orderId << ":" << order.quantity << " "; }
                 std::cout << ")" << std::endl;
            }
        }
        std::cout << "--- End Order Book ---" << std::endl;
    }
};


// --- Order Generator Function (from previous example) ---
std::vector<Order> generateSampleOrders(int numberOfOrders) {
    std::vector<Order> orders;
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> sideDist(0, 1);
    std::uniform_int_distribution<int> typeDist(0, 4); // Give Market orders a chance
    std::uniform_real_distribution<double> priceDist(99.50, 100.50); // Tighter spread
    std::uniform_int_distribution<int> qtyDist(10, 50);
    long long currentOrderId = 1000;

    orders.reserve(numberOfOrders);

    for (int i = 0; i < numberOfOrders; ++i) {
        OrderSide side = (sideDist(rng) == 0) ? OrderSide::BUY : OrderSide::SELL;
        // Make ~20% market orders
        OrderType type = (typeDist(rng) == 0) ? OrderType::MARKET : OrderType::LIMIT;
        double price = (type == OrderType::LIMIT) ? round(priceDist(rng) * 100.0) / 100.0 : 0.0;
        int quantity = qtyDist(rng);

        orders.emplace_back(currentOrderId++, side, type, price, quantity);
         // Small delay to ensure unique timestamps if needed, but can slow down generation
         // std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    return orders;
}

// --- Main Function (Updated Example Usage) ---

int main() {
    OrderBook book; // Create an instance of the OrderBook

    std::cout << "--- Generating Sample Orders ---" << std::endl;
    int numOrdersToGenerate = 20; // Generate more orders
    std::vector<Order> incomingOrders = generateSampleOrders(numOrdersToGenerate);

    std::cout << "\n--- Processing Incoming Orders (Adding to Book) ---" << std::endl;

    // Process generated orders and add LIMIT orders to the book
    for (const auto& order : incomingOrders) {
        std::cout << "Processing: ";
        order.print(); // Print the order being processed
        std::cout << std::endl;

        if (order.type == OrderType::LIMIT) {
            book.addOrder(order); // Add limit orders to the book
        } else {
             std::cout << "Market Order ID " << order.orderId << " received (will be handled by matching engine)." << std::endl;
            // TODO: In the next step, pass MARKET orders (and new LIMIT orders)
            // to the matchingEngine.processOrder(order, book);
        }
         // Print the book state after adding each limit order (for debugging)
         // book.printBook();
         // std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Slow down for viz
    }

    std::cout << "\n--- Final Order Book State ---" << std::endl;
    book.printBook(); // Print the final state of the book

    // Example of checking top of book
    std::cout << "\n--- Top of Book ---" << std::endl;
    std::cout << "Best Bid: " << std::fixed << std::setprecision(2) << book.getBestBid() << std::endl;
    std::cout << "Best Ask: " << std::fixed << std::setprecision(2) << book.getBestAsk() << std::endl;

     // Example of removing an order (using the inefficient search for now)
     std::cout << "\n--- Testing Order Removal ---" << std::endl;
     long long orderIdToRemove = 1005; // Example ID, might not exist if it was market/invalid
     if (book.removeOrder(orderIdToRemove)) {
         std::cout << "Successfully removed order " << orderIdToRemove << std::endl;
         book.printBook();
     } else {
         std::cout << "Order " << orderIdToRemove << " not found or not removable." << std::endl;
     }


    std::cout << "\n--- Finished ---" << std::endl;

    return 0;
}
