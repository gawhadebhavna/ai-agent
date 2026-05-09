ENTITY_REGISTRY = {

    "customer": {

        "object_name": "business_partner",

        "metadata_endpoint":
        "/sap/opu/odata/sap/API_BUSINESS_PARTNER/$metadata",

        "data_endpoint":
        "/sap/opu/odata/sap/API_BUSINESS_PARTNER/A_BusinessPartner"
    },

    "product": {

        "object_name": "product",

        "metadata_endpoint":
        "/sap/opu/odata/sap/API_PRODUCT_SRV/$metadata",

        "data_endpoint":
        "/sap/opu/odata/sap/API_PRODUCT_SRV/A_Product"
    },

    "sales_order": {

        "object_name": "sales_order",

        "metadata_endpoint":
        "/sap/opu/odata/sap/API_SALES_ORDER_SRV/$metadata",

        "data_endpoint":
        "/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder"
    }
}