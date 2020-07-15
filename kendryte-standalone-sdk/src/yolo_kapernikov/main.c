#include "board_config.h"
#include "bsp.h"
#include "dvp.h"
#include "fpioa.h"
#include "gpiohs.h"
#include "image_process.h"
#include "kpu.h"
#include "lcd.h"
#include "nt35310.h"
#include "ov2640.h"
#include "ov5640.h"
#include "plic.h"
#include "prior.h"
#include "yolo_region_layer.h"
#include "sysctl.h"
#include "uarths.h"
#include "utils.h"
#include "w25qxx.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
#include "iomem.h"
#include "utils.h"

    #define PLL0_OUTPUT_FREQ 800000000UL
    #define PLL1_OUTPUT_FREQ 400000000UL

#define CLASS_NUMBER 20

volatile uint32_t g_ai_done_flag;
volatile uint8_t g_dvp_finish_flag;

static image_t kpu_image, display_image;
//static region_layer_t rl;

//#define KMODEL_SIZE (412 * 1024)
#define KMODEL_SIZE (3850 * 1024)
uint8_t model_data[KMODEL_SIZE];

kpu_model_context_t task;

static int dvp_irq(void *ctx) {
    if (dvp_get_interrupt(DVP_STS_FRAME_FINISH)) {
        dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 0);
        dvp_clear_interrupt(DVP_STS_FRAME_FINISH);
        g_dvp_finish_flag= 1;
    } else {
        dvp_start_convert();
        dvp_clear_interrupt(DVP_STS_FRAME_START);
    }
    return 0;
}

static void io_mux_init(void) {
//#if BOARD_LICHEEDAN
    /* Init DVP IO map and function settings */
    fpioa_set_function(42, FUNC_CMOS_RST);
    fpioa_set_function(44, FUNC_CMOS_PWDN);
    fpioa_set_function(46, FUNC_CMOS_XCLK);
    fpioa_set_function(43, FUNC_CMOS_VSYNC);
    fpioa_set_function(45, FUNC_CMOS_HREF);
    fpioa_set_function(47, FUNC_CMOS_PCLK);
    fpioa_set_function(41, FUNC_SCCB_SCLK);
    fpioa_set_function(40, FUNC_SCCB_SDA);

    /* Init SPI IO map and function settings */
    fpioa_set_function(38, FUNC_GPIOHS0 + DCX_GPIONUM);
    fpioa_set_function(36, FUNC_SPI0_SS3);
    fpioa_set_function(39, FUNC_SPI0_SCLK);
    fpioa_set_function(37, FUNC_GPIOHS0 + RST_GPIONUM);

    sysctl_set_spi0_dvp_data(1);
}

static void io_set_power(void) {
//#if BOARD_LICHEEDAN
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK6, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK7, SYSCTL_POWER_V18);
}

#if (CLASS_NUMBER > 1)

typedef struct {
    char *str;
    uint16_t color;
    uint16_t height;
    uint16_t width;
    uint32_t *ptr;
} class_lable_t;

class_lable_t class_lable[CLASS_NUMBER]= {
        {"aeroplane", GREEN}, {"bicycle", GREEN},   {"bird", GREEN},        {"boat", GREEN},
        {"bottle", 0xF81F},   {"bus", GREEN},       {"car", GREEN},         {"cat", GREEN},
        {"chair", 0xFD20},    {"cow", GREEN},       {"diningtable", GREEN}, {"dog", GREEN},
        {"horse", GREEN},     {"motorbike", GREEN}, {"person", 0xF800},     {"pottedplant", GREEN},
        {"sheep", GREEN},     {"sofa", GREEN},      {"train", GREEN},       {"tvmonitor", 0xF9B6}};

static uint32_t lable_string_draw_ram[115 * 16 * 8 / 2];
#endif

static void lable_init(void) {
#if (CLASS_NUMBER > 1)
    uint8_t index;

    class_lable[0].height= 16;
    class_lable[0].width= 8 * strlen(class_lable[0].str);
    class_lable[0].ptr= lable_string_draw_ram;
    lcd_ram_draw_string(class_lable[0].str, class_lable[0].ptr, BLACK, class_lable[0].color);
    for (index= 1; index < CLASS_NUMBER; index++) {
        class_lable[index].height= 16;
        class_lable[index].width= 8 * strlen(class_lable[index].str);
        class_lable[index].ptr= class_lable[index - 1].ptr +
                                class_lable[index - 1].height * class_lable[index - 1].width / 2;
        lcd_ram_draw_string(class_lable[index].str, class_lable[index].ptr, BLACK,
                            class_lable[index].color);
    }
#endif
}

static void drawboxes(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t class,
                      float prob, uint32_t *landmark, uint32_t landm_num) {
    if (x1 >= 320) x1= 319;
    if (x2 >= 320) x2= 319;
    if (y1 >= 224) y1= 223;
    if (y2 >= 224) y2= 223;

#if (CLASS_NUMBER > 1)
    lcd_draw_rectangle(x1, y1, x2, y2, 2, class_lable[class].color);
    lcd_draw_picture(x1 + 1, y1 + 1, class_lable[class].width, class_lable[class].height,
                     class_lable[class].ptr);
#else
    lcd_draw_rectangle(x1, y1, x2, y2, 2, RED);
    for (uint32_t i= 0; i < landm_num; i++) {
        lcd_draw_point(landmark[2 * i], landmark[1 + 2 * i], GREEN);
    }
#endif
}

static int index_max(float *a, int n) {
    int i, max_i= 0;
    float max= a[0];

    for (i= 1; i < n; ++i) {
        if (a[i] > max) {
            max= a[i];
            max_i= i;
        }
    }
    return max_i;
}


void print_boxes(region_layer_t *rl) {
    uint32_t image_width= rl->image_width;
    uint32_t image_height= rl->image_height;
    float threshold= rl->threshold;
    box_t *boxes= (box_t *)rl->boxes;

    for (int i= 0; i < rl->boxes_number; ++i) {
        int class= index_max(rl->probs[i], rl->classes);
        float prob= rl->probs[i][class];

        if (prob > threshold) {
            box_t *b= boxes + i;
            uint32_t x1= b->x * image_width - (b->w * image_width / 2);
            uint32_t y1= b->y * image_height - (b->h * image_height / 2);
            uint32_t x2= b->x * image_width + (b->w * image_width / 2);
            uint32_t y2= b->y * image_height + (b->h * image_height / 2);
            printf("(x1, y1), (x2, y2) -- (%i, %i), (%i, %i)\n", x1, y1, x2, y2);
            printf("class:  ");
            printf(class_lable[i].str);
            printf("\n");
        }
    }
}

static int ai_done(void *ctx) {
    g_ai_done_flag= 1;
    return 0;
}

#define ANCHOR_NUM 3

static region_layer_t detect_rl0, detect_rl1;

//static float layer0_anchor[ANCHOR_NUM * 2]= {
//        0.76120044, 0.57155991, 0.6923348, 0.88535553, 0.47163042, 0.34163313,
//};
//
//static float layer1_anchor[ANCHOR_NUM * 2]= {
//        0.33340788, 0.70065861, 0.18124964, 0.38986752, 0.08497349, 0.1527057,
//};

float g_anchor[ANCHOR_NUM * 2] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};



int main(void) {
    /* Set CPU and dvp clk */
    sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    uarths_init();
    io_set_power();
    io_mux_init();
    plic_init();

     lable_init();

//    Load model from flash memory
    printf("flash init\n");
    w25qxx_init(3, 0);
    w25qxx_enable_quad_mode();
    w25qxx_read_data(0xA00000, model_data, KMODEL_SIZE, W25QXX_QUAD_FAST);

    /* LCD init */
    printf("LCD init\n");
    lcd_init();

    lcd_set_direction(DIR_YX_RLDU);

    lcd_clear(BLACK);

    /* DVP init */
    printf("DVP init\n");
    dvp_init(8);
    dvp_set_xclk_rate(24000000);
    dvp_enable_burst();
    // DVP can output to both KPU and main mem, set output mode for both KPU (index 0) and main mem (index 1) to enablded (1)
    dvp_set_output_enable(0, 1);
    dvp_set_output_enable(1, 1);

    dvp_set_image_format(DVP_CFG_RGB_FORMAT);
    // set_image_size(width, height)
    dvp_set_image_size(320, 224);
    ov2640_init();

    // initialize memory for KPU (?)
    kpu_image.pixel= 3;
    kpu_image.width= 320;
    kpu_image.height= 224;
    image_init(&kpu_image);

    // initialize memory for display image
    display_image.pixel= 2;
    display_image.width= 320;
    display_image.height= 224;
    image_init(&display_image);

    // DVP- Digital Video Port - can forward camera input to both KPU and memory

    // "Set the image address required by the KPU, in order to facilitate the KPU to perform algorithm processing."
    dvp_set_ai_addr((uint32_t)kpu_image.addr, (uint32_t)(kpu_image.addr + 320 * 240),
                (uint32_t)(kpu_image.addr + 320 * 240 * 2));

    // "Set the storage address of the captured image in the memory, which can be used for display."
    dvp_set_display_addr((uint32_t)display_image.addr);

    dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 0);
    dvp_disable_auto();

    /* DVP interrupt config */
    printf("DVP interrupt config\n");
    plic_set_priority(IRQN_DVP_INTERRUPT, 1);
    plic_irq_register(IRQN_DVP_INTERRUPT, dvp_irq, NULL);
    plic_irq_enable(IRQN_DVP_INTERRUPT);

    if (kpu_load_kmodel(&task, model_data) != 0) {
        printf("\nmodel init error\n");
        while (1) {};
    }


    // detect_rl0, detect_rl0 contain information about /configuration for the region layer
    // will also contain pointers to results of the region layer
    detect_rl0.anchor_number= ANCHOR_NUM;
    detect_rl0.anchor= g_anchor;
//    detect_rl0.threshold= 0.6;
//    detect_rl0.nms_value= 0.3;
    detect_rl0.threshold= 0.5;
    detect_rl0.nms_value= 0.2;
    region_layer_init(&detect_rl0, 10, 7, 125, 320, 240);    // region_layer_t *rl, int width, int height, int channels, int origin_width,int origin_height

//    detect_rl1.anchor_number= ANCHOR_NUM;
//    detect_rl1.anchor= layer1_anchor;
//    detect_rl1.threshold= 0.6;
//    detect_rl1.nms_value= 0.3;
//    region_layer_init(&detect_rl1, 20, 14, 75, 320, 240);


    /* enable global interrupt */
    sysctl_enable_irq();
    /* system start */
    printf("System start\n");

    float *output0, *output1;
    size_t output_size0, output_size1;
    clock_t start, end;
    double runtime;

    float *pred_box, *pred_landm, *pred_clses;
    size_t pred_box_size, pred_landm_size, pred_clses_size;


    while (1) {
        g_dvp_finish_flag= 0;
        dvp_clear_interrupt(DVP_STS_FRAME_START | DVP_STS_FRAME_FINISH);
        dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 1);
        while (g_dvp_finish_flag == 0) {};

        /* run face detect */
        g_ai_done_flag= 0;

        start = clock();
        kpu_run_kmodel(&task, (uint8_t *)kpu_image.addr, DMAC_CHANNEL5, ai_done, NULL);
        // wait for it to be done
        while (!g_ai_done_flag) {};

        //store model output in output0, output1, and store output size
        kpu_get_output(&task, 0, (uint8_t **)&output0, &output_size0);
//        kpu_get_output(&task, 1, (uint8_t **)&output1, &output_size1);

        detect_rl0.input= output0;
        region_layer_run(&detect_rl0, NULL);
//        detect_rl1.input= output1;
//        region_layer_run(&detect_rl1, NULL);


//        /* run key point detect */
        printf("detected boxes rl0 --  \n");
        print_boxes(&detect_rl0);
//        printf("detected boxes rl1 --  \n");
//        print_boxes(&detect_rl1);
        lcd_draw_picture(0, 0, 320, 224,  (uint32_t *)display_image.addr);

        // draw boxes
        region_layer_draw_boxes(&detect_rl0, drawboxes);
//        region_layer_draw_boxes(&detect_rl1, drawboxes);

        end = clock();
        runtime = (double)(end  - start) / CLOCKS_PER_SEC;
        printf("runtime: %f \n",runtime);
    }
}
